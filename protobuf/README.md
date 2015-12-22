# Protocol buffer schema

The `.proto` file defines the data schema for LOPQ model parameters.
It can be compiled into libraries for any target languages
that will use this data (i.e., Java and Python). A compiled version
for Python is included in the `lopq` module.

See: [https://developers.google.com/protocol-buffers/docs/overview](https://developers.google.com/protocol-buffers/docs/overview)
for details on protocol buffers, and how to update the schema.
Note, to keep the schema backwards compatable, it is important not to
alter the `tags` (ints) assigned to each field in `.proto` file.


### Compiling `.proto` file

Compile for Java:

```bash
# from the repository root
protoc -I=protobuf \
       --java_out=. \
       protobuf/lopq_model.proto
```

Compile for Python:

```bash
# from the repository root
protoc -I=protobuf \
       --python_out=python/lopq \
       protobuf/lopq_model.proto
```
