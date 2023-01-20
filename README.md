# Graph ML on PyTorch Geometric in Neo4j

[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) is a Python library for dealing with graph algorithms.

## Installation

Use the package manager [poetry](https://python-poetry.org/) in myenv to install foobar. Install [pyenv](https://github.com/pyenv/pyenv) befrorehand.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install poetry --no-cache   
poetry install
```

## Usage

Unzip `data.zip` file to the main directory and copy contents of `import_bckp` to `import folder` in the same directory.

Once the data is in place run Docker command.

```dockerfile
CONTAINER=$(docker run \                                          
    --name testneo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $(pwd)/data/neo4j_db/data:/data \
    -v $(pwd)/data/neo4j_db/logs:/logs \
    -v $(pwd)/data/neo4j_db/import:/var/lib/neo4j/import \
    -v $(pwd)/data/neo4j_db/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/test_password \
    --env NEO4JLABS_PLUGINS='["graph-data-science", "apoc"]' \
    neo4j:latest)              
```

### Setup
Once Docker Container is up and running, create contents based on queries in `db_loader.cypher` file.
You have few options:
1. _(Easy-mode)_ You can run them in [browser](http://localhost:7474/browser) and copy paster.
2. Within terminal run ->

`$ docker exec $CONTAINER /var/lib/neo4j/bin/neo4j-shell -f /var/lib/neo4j/db_loader.cypher`

or for interactive mode

`$ docker exec -ti $CONTAINER /var/lib/neo4j/bin/neo4j-shell`

Then run following code in the terminal for training model and creating a new `RECOMMENDED_TO` relationship.

```bash
python recommendations/main.py
```

---
Pulling data from Neo4j and loading results to Neo4j are made with the use of `["graph-data-science", "apoc"]` plugins.
Example of new mapping can be found in `results.txt` file, but it is not updated after new training.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Credits
* [Bartosz Mielczarek](https://www.linkedin.com/in/bartosz-mielczarek-647346117)
* [Piotr Walkowski](https://www.linkedin.com/in/piotrwalkowski/)


## License

[MIT](https://choosealicense.com/licenses/mit/)