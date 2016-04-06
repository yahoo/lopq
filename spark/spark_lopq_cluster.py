# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
"""
These classes launch a distributed LOPQ search cluster on Spark using spark-partition-server
(see https://github.com/pumpikano/spark-partition-server for more information).

It expects either an RDD of (id, code) pairs where `code` is a tuple of LOPQ codes, or an
RDD of (id, ndarray) pairs where `ndarray` is a vector to index.

Usage example:
```
# Load a model
model = LOPQModel.load_proto('path/to/model.lopq')

# Create a cluster object with the SparkContext, an RDD in the allowed form, and the model object
c = LOPQCluster(sc, rdd, model)

# Start the cluster - this builds an LOPQSearcher on each partition and launches a server to query it
c.start()

# You can query it directly (or on the prompt) with some query x:
results = c.search(x)

# The LOPQClusterServer allows other clients (like a webpage) to query the cluster:
t = LOPQClusterServer(c, port=33685)
t.start()

# Stop the cluster server and the cluster:
t.stop()
c.stop()
```
"""
import json
from threading import Thread
from itertools import chain, islice
import flask
from flask import Blueprint, request, jsonify
from flask.ext.cors import CORS
import requests
import numpy as np
from spark_partition_server import Cluster, FlaskPartitionServer, ServerThread
from lopq.search import multisequence, LOPQSearcher


class LOPQPartitionServer(FlaskPartitionServer):
    def __init__(self, **kwargs):

        blueprint = Blueprint('app', __name__)
        @blueprint.route('/search', methods=['GET'])
        def search():
            """Execute an LOPQ search on the partition"""
            from flask import request, jsonify
            import numpy as np

            # Expect query param 'vector' to contain a comma-separated string of float values
            x = request.args.get('vector')
            x = np.array(map(float, x.split(',')))

            # 
            cells = request.args.get('cells')
            cells = json.loads(cells)

            limit = request.args.get('limit')
            limit = int(limit)

            items = list(chain(*[self.searcher.get_cell(tuple(cell)) for cell in cells]))
            results = self.searcher.compute_distances(x, items)
            results = sorted(results, key=lambda x: x[0])
            results = results[:limit]

            return jsonify({
                'count': len(results),
                'results': map(lambda (dist, item): { 'dist': dist, 'id': item[0], 'code': item[1] }, results),
                'partition': self.partition_ind
            })

        super(LOPQPartitionServer, self).__init__(blueprint=blueprint, **kwargs)

        if 'codes' not in self.config:
            self.config.update({'codes': True})

    def init_partition(self, itr, app, config):
        """Create an LOPQSearcher for this partition and add the partition data"""
        partition = list(itr)
        itemids, data = zip(*partition)
        self.searcher = LOPQSearcher(config['model_bc'].value)

        if config['codes']:
            self.searcher.add_codes(data, ids=itemids)
        else:
            self.searcher.add_data(data, ids=itemids)


class LOPQCluster(Cluster):
    def __init__(self, sc, rdd, model, config=None, **kwargs):
        """Create a Cluster object to control the partition searchers

        :param SparkContext sc:
        :param RDD rdd:
        :param LOPQModel model:
        :param dict config:
            a config dict to be passed to partition servers; with the 'codes' flag
            you can control whether the RDD is of lopq code tuples are raw vectors

        """
        super(LOPQCluster, self).__init__(sc, rdd, **kwargs)

        self.model = model

        config = { 'codes': True }
        config.update(kwargs.pop('config', {}))
        config.update(model_bc=self.sc.broadcast(self.model))
        self.partition_server = LOPQPartitionServer(config=config)

    def search_partition(self, ind, x, cells, limit=100):
        """Search a single partition

        :param int ind: the partition index
        :param ndarray x: a query vector
        :param cells: a list of cells (tuples of ints) to rank
        :param int limit: the number of results desired
        """
        url = 'http://%s:%d/app/search' % self.coordinator.hosts[ind]
        url = '%s?vector=%s&cells=%s&limit=%d' % (url, ','.join(map(str, x)), json.dumps(cells), limit)

        res = requests.get(url)

        return json.loads(res.text)['results']


    def search(self, x, limit=100, num_cells=10):
        """Search the cluster with a query vector in the first num_cells

        :param ndarray x: a query vector
        :param int limit: the desired number of results
        :param int num_cells: the number of best cells to check
        """
        from Queue import Queue

        # estimate number of results each partition should return
        limit = max(limit / len(self.coordinator.hosts), 1)

        # get list of cells to look in
        m = multisequence(x, self.model.Cs)
        _, cells = zip(*list(islice(m, num_cells)))

        # wrap call to search_partition to execute in thread
        q = Queue()
        def target_fn(*args):
            q.put(self.search_partition(*args))

        threads = []
        for ind in self.coordinator.hosts:
            thread = Thread(target=target_fn, args=(ind, x, cells, limit))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        items = chain(*[q.get() for x in range(q.qsize())])

        return sorted(items, key=lambda item: item['dist'])


class LOPQClusterServer(ServerThread):
    """Launch a Flask server in another thread to relay queries to the cluster"""
    def __init__(self, cluster, port=None):
        """Create a server to front the provided LOPQCluster instance"""
        self.cluster = cluster
        self._build_app()
        super(LOPQClusterServer, self).__init__(self.app, port=port)

    def _build_app(self):
        """Construct a Flask app with a simple /search route"""
        app = flask.Flask('lopq')
        CORS(app, supports_credentials=True)

        @app.route('/search', methods=['GET'])
        def search():
            
            x = request.args.get('lopq.vector')
            x = np.array(map(float, x.split(',')))

            cells = request.args.get('cells', 10)
            cells = int(cells)

            limit = request.args.get('num', 100)
            limit = int(limit)

            results = self.cluster.search(x, limit=limit, num_cells=cells)
            results = results[:limit]

            return jsonify({
                'count': len(results),
                'items': results
            })

        self.app = app
