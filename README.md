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

Now build desired `neo4j` container.
```dockerfile
CONTAINER=$(docker run -d \
    -p 7474:7474 -p 7687:7687 \
    -v $(pwd)/data/neo4j_db/data:/data \
   -v $(pwd)/data/neo4j_db/logs:/logs \
   -v $(pwd)/data/neo4j_db/import:/var/lib/neo4j/import \
    --name test-neo4j-stx-books-recommender44 \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4J_AUTH=neo4j/stx_books_pass \
   -e NEO4JLABS_PLUGINS='["apoc", "graph-data-science"]' \
   -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
   neo4j:4.4-enterprise
)                  
```

*Of note: We use here 4.4 version due to not being stable (at 30.01) APOC version from 5. x. This might vary in future.*

### Setup
Once Docker Container is up and running, create contents based on queries in `YOUR_DOCKER_NEO_LOCATION/db_loader.cypher` file.
You have few options:
1. _(Easy-mode)_ You can run them in [browser](http://localhost:7474/browser) and just copy-paste.
2. Within terminal run ->

`$ docker exec $CONTAINER /var/lib/neo4j/bin/neo4j-shell -f YOUR_DOCKER_NEO_LOCATION/db_loader.cypher`

or for interactive mode... (to copy-paste like in the browser)

`$ docker exec -ti $CONTAINER /var/lib/neo4j/bin/neo4j-shell`

#### Important! (ML Flow setup)

Before running your code, you need to define all variables stored in `.env`.

Especially:
```commandline
MLFLOW_USER=
MLFLOW_PASSWORD=
MLFLOW_URL=
```

So either uses your own **MLFlow** account or use your dockerized one.

---
### Data schema

After proper data population within the graph database there should be visible following schema:
![Database Schema](assets/img/schemat.png)
Or you can try by yourself by calling 
```cypher 
CALL db.schema.visualization()
```
#### So we have the following:
* **Users** - representing our users with some attributes (including `first_name`, `last_name` etc)
* **Titles** - representing specific books with their metadata. Connected with a user with relations `RATED_BY` and `READ_BY`. While `RATED_BY` has its wage (0-10) and is used for further embeddings via FastRB to classify and obtain our recommendations (that will be modelled via `RECOMMENDED_BY`)
* **Authors** - Node that points to the given Author of the book, with its metadata. By relation `WRITTEN_BY`
* **YearsOfPublications** - node for a specific year of publication (via `WRITTEN_IN_YEAR` relation)
* **Publishers** - node representing publisher of a given book (via `PUBLISHED_BY` relation)


More detailed schema (with specific indices in `csv` view) can be read [here](assets/full_schema_prior_mapping.csv) 

### Dataset source
Our dataset comes from [Kaggle Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
It was modified limited to 50k and for readability by adding fixtures to *Users* (first name, last name) by [faker](https://github.com/joke2k/faker) so that *any similarity to real person is pure coincidence* :)


### Running
Then run the following code in the terminal for the training model and create a new `RECOMMENDED_TO` relationship. 

```bash
python3 main.py
```


Obviously, the relationship is between `Titles` and `Users`
`(Titles)-[:RECOMMENDED_TO)->(Users)`

Below is a fracture of new relationships:
![Example recommendations](assets/img/recommendation_results.png)

How the process of embeddings (to temporary `book_titles` graph) looks like:
![FastRP embeeding](assets/img/fast_rp_embedding.png)


---

## Examples 
```diff
- TODO: See also our blog-post!
```

- [link to STX blogpost here](https://docs.google.com/document/d/1AJ7Gm_Lc-wcRb_IXm1GvShT6c2yQ8Q0csD_6vLwYDLA/edit#) for more (TODO: or copy-paste here)

Graph-based recommendations give us *a very powerful* tool to search by different criteria. Where our imagination is the limit.

* ### Finding recommendations based on User criteria

Results of recommendation for a specific user (in this case Patti Jacobs)
![All recommendations to user "Patti Jacobs"](assets/img/patti_jacobs_recommendations.png)
```cypher
MATCH paths=(u: Users {first_name: 'Patti', last_name: 'Jacobs'})-[:RECOMMENDED_TO]->(t:Titles) RETURN paths;
```

* ### Overlapping sets 

List of readers that loves *"pride & prejudice"* to check what they have in common:
![Recommendation based on most popular](assets/img/pride_1.png)
For [results CSV](assets/pride_recommendations.csv) 

```cypher
MATCH (romance_lovers:Users)-[:READ_BY]->(n:Titles) WHERE n.title = 'Pride and Prejudice'
MATCH (other_book:Titles)-[:RECOMMENDED_T0]->(romance_lovers:Users)
WHERE id(other_book) <> id(n)
RETURN other_book.author AS author, other_book.title AS title;
```

* ### Recommendations based on popularity

What are the best guesses for top-5 book readers?
Below the cypher, sub-query obtaining first part 
![Most popular - cypher](assets/img/most_popular_cypher.png)

Full query showing all recommendations
![Most popular - graph](assets/img/top_based_on_most_popular.png)
For [results CSV](assets/most_popular_books_by_reads.csv)

```cypher
CALL {
  MATCH (users:Users)-[:READ_BY]->(n:Titles)
  WITH COUNT(n) AS counter, n, COLLECT(id(users)) AS user_ids
  RETURN n.title, counter, user_ids
  ORDER BY counter DESC
  LIMIT 5
}
WITH user_ids
UNWIND user_ids AS user_id
MATCH (u:Users {user:user_id})-[:RECOMMENDED_TO]->(t2:Titles)
RETURN t2
LIMIT 10;
```

* ### Recommendation on any criteria

Here make the limitation to only readers based on `US` that have already rated books published after `1984`!

![Recommended after 84'](assets/img/recommended_us_after_1984.png)
For [results CSV](assets/recommended_after_84.csv)

```cypher
CALL {
  MATCH (u:Users)-[r:RATED_BY]->(t:Titles)
  WITH lTrim(split(u. location, ',')[-1]) AS location, t, u
  WHERE Location - 'usa' AND t.year_of_publication > 1984
  RETURN t, u
  LIMIT 10
}
WITH u
MATCH (u)<-[:RECOMMENDED_TO]-(t2:Titles)
RETURN t2.author AS recommended_author, t2.title AS recommended_title
LIMIT 5;
```

## Next steps
Pulling data from Neo4j and loading results to Neo4j are made with the use of `["graph-data-science", "apoc"]` plugins.

For a visualisation - an example of new mapping can be found in the `sample/results.txt` file, but it is not updated after new training.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Credits
* [Bartosz Mielczarek](https://www.linkedin.com/in/bartosz-mielczarek-647346117)
* [Piotr Walkowski](https://www.linkedin.com/in/piotrwalkowski/)


## License

[MIT](https://choosealicense.com/licenses/mit/)
