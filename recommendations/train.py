import torch

from loguru import logger
from tqdm import tqdm

import torch.nn.functional as F

import pytorch_lightning as pl

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit

from recommendations import DEVICE
from recommendations.consts import (
    ENCODER_MODEL_NAME, TRAIN_FRAC, VALID_FRAC, TEST_FRAC, NEG_SAMPLING_RATIO, HIDDEN_CHANNELS, LEARNING_RATE, EPOCHS,
    MIN_PRED_VALUE, MAX_PRED_VALUE, PRED_BENCHMARK, MAX_PRED_USERS, MAX_PRED_RECOMMENDATIONS,
    MLFLOW_TRACKING_PATH, MLFLOW_EXPERIMENT_NAME
)
from recommendations.models import Model


class RecommendationsOnGraph:
    """
    Rebuild locally Heterogeneous Graph in Pytorch Geometric and build Graph Edges for Recommendation.
    https://pytorch-geometric.readthedocs.io/en/latest/
    """

    def __init__(self, data_dict: dict) -> None:
        self.data_dict = data_dict


    def _build_heterogeneous_graph(self, data_dict: dict):
        """
        Add features to graph.
        """
        data = HeteroData()
        # Add user node features for message passing:
        data['user'].x = torch.eye(len(data_dict["mapping"]["user"]), device=DEVICE)
        # Add movie node features
        data['title'].x = data_dict["x"]["title"]
        # Add ratings between users and movies
        data['user', 'rates', 'title'].edge_index = data_dict["edge_index"]["rating"]
        data['user', 'rates', 'title'].edge_label = data_dict["edge_label"]["rating"]
        data.to(DEVICE, non_blocking=True)

        # Transform to undirected graph.
        data = ToUndirected()(data)
        # Remove "reverse" label.
        del data['title', 'rev_rates', 'user'].edge_label

        return data


    def _weighted_mse_loss(self, pred, target, weight=None):
        """
        MSE Loss definition
        """
        weight = 1. if weight is None else weight[target].to(pred.dtype)
        return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


    def _train_valid_test_split(self, data):
        """
        Split data (by link) to training, validation and test sets.
        """
        transform = RandomLinkSplit(
            num_val=VALID_FRAC,
            num_test=TEST_FRAC,
            neg_sampling_ratio=NEG_SAMPLING_RATIO,
            edge_types=[('user', 'rates', 'title')],
            rev_edge_types=[('title', 'rev_rates', 'user')],
        )
        return transform(data)


    def _train(self, model, optimizer, train_data, weight):
        """
        Training Model Function
        """
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict,
                     train_data['user', 'rates', 'title'].edge_label_index)
        target = train_data['user', 'rates', 'title'].edge_label
        loss = self._weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def _test(self, data, model):
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict,
                     data['user', 'rates', 'title'].edge_label_index)
        pred = pred.clamp(min=MIN_PRED_VALUE, max=MAX_PRED_VALUE)
        target = data['user', 'rates', 'title'].edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        return float(rmse)

    def _print_auto_logged_info(self, r):
        """
        Local logger for MLFlow artifacts.
        """
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))


    def _run_model(self):
        """
        Train Selected Graph Model, set mlflow connection and
        """
        data = self._build_heterogeneous_graph(self.data_dict)
        (train_data, val_data, test_data) = self._train_valid_test_split(data=data)
        weight = torch.bincount(train_data['user', 'title'].edge_label)
        weight = weight.max() / weight

        # MLFlow logging
        mlflow.set_tracking_uri(MLFLOW_TRACKING_PATH)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Initialize the model
        model = Model(hidden_channels=HIDDEN_CHANNELS, metadata=data.metadata()).to(DEVICE)
        with torch.no_grad():
            model.encoder(train_data.x_dict, train_data.edge_index_dict)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # ----------------- VERSION BASIC -----------------
        # Train the model
        with mlflow.start_run() as run:
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("model_architecture", "GNNEncoder -> to_hetero -> EdgeDecoder")
            mlflow.log_param("encoder_model_name", ENCODER_MODEL_NAME)
            mlflow.log_param("train_fraction", TRAIN_FRAC)
            mlflow.log_param("valid_fraction", VALID_FRAC)
            mlflow.log_param("test_fraction", TEST_FRAC)
            mlflow.log_param("neg_sampling_ratio", NEG_SAMPLING_RATIO)
            mlflow.log_param("hidden_channels", HIDDEN_CHANNELS)
            mlflow.log_param("learning_rate", LEARNING_RATE)
            for epoch in range(1, EPOCHS):
                # with mlflow.start_run(nested=True):
                loss = self._train(model=model, optimizer=optimizer, train_data=train_data, weight=weight)
                train_rmse = self._test(data=train_data, model=model)
                val_rmse = self._test(data=val_data, model=model)
                test_rmse = self._test(data=test_data, model=model)
                logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')

                mlflow.log_metric("loss", loss)
                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("val_rmse", val_rmse)
                mlflow.log_metric("test_rmse", test_rmse)

            mlflow.pytorch.log_model(model, "book_recommendations_gnn_encoder_model",
                                     registered_model_name="BookRecommendationsGNNEncoderModel")


        # ----------------- VERSION ADVANCED -----------------
        # TODO: Add AutoLog Based on the Pytorch Lightning
        # # Auto log all MLflow entities
        # mlflow.pytorch.autolog()
        #
        # # Initialize trainer
        # trainer = pl.Trainer(max_epochs=20, progress_bar_refresh_rate=20)
        #
        # # Train the model
        # with mlflow.start_run() as run:
        #     trainer.fit(model, train_loader)
        #
        # # fetch the auto logged parameters and metrics
        # self._print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

        return data, model


    def generate_predictions(self):
        """
        Generates recommendations based on predictions from the model.
        """
        title_mapping = self.data_dict["mapping"]["title"]
        user_mapping = self.data_dict["mapping"]["title"]
        data, model = self._run_model()

        num_title = len(title_mapping)
        num_users = len(user_mapping)

        reverse_title_mapping = dict(zip(title_mapping.values(),title_mapping.keys()))
        reverse_user_mapping = dict(zip(user_mapping.values(),user_mapping.keys()))

        recommenations_pred = []

        for user_id in tqdm(range(0,MAX_PRED_USERS)):

            row = torch.tensor([user_id] * num_title)
            col = torch.arange(num_title)
            edge_label_index = torch.stack([row, col], dim=0)
            pred = model(data.x_dict, data.edge_index_dict,
                         edge_label_index)
            pred = pred.clamp(min=MIN_PRED_VALUE, max=MAX_PRED_VALUE)

            user_neo4j_id = reverse_user_mapping[user_id]

            mask = (pred > PRED_BENCHMARK).nonzero(as_tuple=True)

            ten_predictions = [reverse_title_mapping[el] for el in  mask[0].tolist()[:MAX_PRED_RECOMMENDATIONS]]
            recommenations_pred.append({'user': user_neo4j_id, 'title': ten_predictions})

        return recommenations_pred