import pandas as pd
import torch

from neo4j import GraphDatabase
from recommendations import GDB_URL, GBD_PORT, GDB_USER, GDB_PASSWORD

from recommendations.encoders import SequenceEncoder, LabelsEncoder, IdentityEncoder


class GraphDBDriver:
    """
    Driver for running queries in neo4j database.
    """

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(f"{GDB_URL}:{GBD_PORT}", auth=(GDB_USER, GDB_PASSWORD))

    def fetch_data(self, query: str, params: dict = {}) -> pd.DataFrame:
        with self.driver.session() as session:
            result = session.run(query, params)
            return pd.DataFrame([r.values() for r in result], columns = result.keys())

    def load_node(self, cypher_query: str, index_col: str, encoders: dict[str, SequenceEncoder | LabelsEncoder | IdentityEncoder] = None):
        # Execute the cypher query and retrieve data from Neo4j
        df = self.fetch_data(cypher_query)
        df.set_index(index_col, inplace=True)
        # Define node mapping
        mapping = {index: i for i, index in enumerate(df.index.unique())}
        # Define node features
        x = None
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)

        return x, mapping

    def load_edge(self, cypher_query: str, src_index_col: str, src_mapping, dst_index_col: str, dst_mapping,
                  encoders=None):
        # Execute the cypher query and retrieve data from Neo4j
        df = self.fetch_data(cypher_query)
        # Define edge index
        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])
        # Define edge features
        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr
