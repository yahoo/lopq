#!/bin/bash

# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

set -e
set -x

# run tests
nosetests -v -s test/tests.py

# make sure we can distribute the module
python setup.py sdist

