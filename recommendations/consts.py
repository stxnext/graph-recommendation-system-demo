from recommendations import MLFLOW_URL_PREFIX, MLFLOW_URL, MLFLOW_PORT, MLFLOW_USER, MLFLOW_PASSWORD

ENCODER_MODEL_NAME = "all-MiniLM-L6-v2"
TRAIN_FRAC = 0.8
VALID_FRAC = 0.1
TEST_FRAC = 0.1
NEG_SAMPLING_RATIO = 0.0
HIDDEN_CHANNELS = 64
LEARNING_RATE = 0.01
EPOCHS = 100
MIN_PRED_VALUE = 1
MAX_PRED_VALUE = 10
PRED_BENCHMARK = 9
MAX_PRED_USERS = 100
MAX_PRED_RECOMMENDATIONS = 10

MLFLOW_TRACKING_PATH = f"{MLFLOW_URL_PREFIX}://{MLFLOW_USER}:{MLFLOW_PASSWORD}@{MLFLOW_URL}:{MLFLOW_PORT}"
MLFLOW_EXPERIMENT_NAME = "book-recommendations-in-graph"

SELECTED_GRAPH = "book_titles"
EMBEDDING_DIMENSION = 56
EMBEDDING = "fastrp"
QUERIES = {
    "list_named_graphs": """
        CALL gds.graph.list()
    """,
    "delete_database": {
        "book_titles": """
            CALL gds.graph.drop('book_titles')
        """
    },
    "create_database": {
        "book_titles": """
            CALL gds.graph.project('book_titles', ['Titles', 'Users', 'Authors', 'Publishers', 'YearsOfPublication'],
              {
                RATED_BY: {orientation:'NATURAL'},
                READ_BY: {orientation:'NATURAL'},
                PUBLISHED_BY: {orientation:'NATURAL'},
                WRITTEN_BY: {orientation:'NATURAL'},
                WRITTEN_IN_YEAR: {orientation:'UNDIRECTED'}
              }
            )
        """
    },
    "create_node_embeddings_in_database": {
        "book_titles": """
            CALL gds.fastRP.write(
                'book_titles', 
                {{
                    writeProperty:{writeProperty}, 
                    embeddingDimension:{embeddingDimension}
                }}
            )
        """.format(writeProperty=EMBEDDING,embeddingDimension=EMBEDDING_DIMENSION)
    },
    "fetch_data_from_database": {
        "book_titles": {
            "user": """
                MATCH (u:Users) RETURN u.user AS user, u.location AS location
            """,
            "title": """
                MATCH (p:Publishers)-[:PUBLISHED_BY]->(t:Titles)
                WITH t, collect(p.publisher) AS publisher_list
                RETURN t.isbn AS isbn, t.title AS title, apoc.text.join(publisher_list, '|') AS publishers, t.fastrp AS fastrp
            """,
            "rating": """
                MATCH (u:Users)-[r:RATED_BY]->(t:Titles)
                RETURN t.isbn AS isbn, u.user AS user, t.title AS title, r.rating AS rating
            """
        }
    },
    "export_data_to_database": {
        "book_titles": {
            "recommended_to": """
                UNWIND $data AS row
                MATCH (u:Users {user: row.user})
                WITH u, row
                UNWIND row.title AS isbn
                MATCH (t:Titles {isbn: isbn})
                WITH u,t
                // filter out existing links
                WHERE NOT (u)-[:RATED_BY]->(t)
                MERGE (t)-[:RECOMMENDED_TO]->(u)
            """
        }
    }
}
