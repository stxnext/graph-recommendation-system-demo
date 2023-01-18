import torch
from loguru import logger

from recommendations.conn import GraphDBDriver
from recommendations.consts import QUERIES, SELECTED_GRAPH
from recommendations.encoders import SequenceEncoder, LabelsEncoder, IdentityEncoder
from recommendations.train import RecommendationsOnGraph


def main():
    logger.info("Get Driver to GraphDB")
    gdb_driver = GraphDBDriver()

    logger.info("Make a new entry to GDS graph list")
    try:
        gdb_driver.fetch_data(query=QUERIES["create_database"][SELECTED_GRAPH])
        gdb_driver.fetch_data(query=QUERIES["create_node_embeddings_in_database"][SELECTED_GRAPH])
    except:
        logger.info("Entry already exists")
        ng_df = gdb_driver.fetch_data(query=QUERIES["list_named_graphs"])["graphName"]
        logger.info(ng_df.tolist())

    logger.info("Fetch Node Data")
    user_x, user_mapping = gdb_driver.load_node(
        cypher_query=QUERIES["fetch_data_from_database"][SELECTED_GRAPH]['user'],
        index_col='user'
    )
    location_x, location_mapping = gdb_driver.load_node(
        cypher_query=QUERIES["fetch_data_from_database"][SELECTED_GRAPH]['user'],
        index_col='location'
    )
    title_x, title_mapping = gdb_driver.load_node(
        cypher_query=QUERIES["fetch_data_from_database"][SELECTED_GRAPH]['title'],
        index_col='isbn', encoders={
            'title': SequenceEncoder(),
            'publishers': LabelsEncoder(),
            'fastrp': IdentityEncoder(is_list=True)
        }
    )

    logger.info("Fetch Edge Data")
    rating_edge_index, rating_edge_label = gdb_driver.load_edge(
        cypher_query=QUERIES["fetch_data_from_database"][SELECTED_GRAPH]['rating'],
        src_index_col='user',
        src_mapping=user_mapping,
        dst_index_col='isbn',
        dst_mapping=title_mapping,
        encoders={'rating': IdentityEncoder(dtype=torch.long)},
    )

    logger.info("Create data dictionary")
    data_dict = {
        "x": {
            "user": user_x,
            "location": location_x,
            "title": title_x
        },
        "mapping":{
            "user": user_mapping,
            "location": location_mapping,
            "title": title_mapping
        },
        "edge_index": {
            "rating": rating_edge_index
        },
        "edge_label": {
            "rating": rating_edge_label
        }
    }

    logger.info("Train Model")
    recommendation_on_graph = RecommendationsOnGraph(data_dict=data_dict)
    recommenations_pred = recommendation_on_graph.generate_predictions()

    logger.info("Export recommendations to Graph DB")
    gdb_driver.fetch_data(
        query=QUERIES["export_data_to_database"][SELECTED_GRAPH]["recommended_to"],
        params={'data': recommenations_pred}
    )


if __name__ == "__main__":
    main()