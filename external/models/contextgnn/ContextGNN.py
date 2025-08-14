"""
Disclaimer: A substantial portion of the code contained in the current package is directly taken from
this public GitHub repository: https://github.com/kumo-ai/ContextGNN/tree/xinwei_add_static_data_and_model_v1 and
adapted to work within the framework Elliot (https://elliot.readthedocs.io/en/latest/).
Please refer to the above cited GitHub repository and to the original paper
of ContextGNN (https://arxiv.org/abs/2411.19513) for further details.
"""

from tqdm import tqdm
import torch
import os
import psutil

from operator import itemgetter

from elliot.utils.write import store_recommendation
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .ContextGNNModel import ContextGNNModel
from .utils import RHSEmbeddingMode
from typing import Any, Dict, Tuple
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_frame.data.stats import StatType
from relbench.modeling.graph import LinkTrainTableInput
from relbench.modeling.loader import SparseTensor
from torch_geometric.seed import seed_everything
from torch_geometric.utils import sort_edge_index
from relbench.modeling.utils import to_unix_time
from torch_frame.utils import infer_df_stype
from torch import Tensor
from torch_frame import stype
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from torch_frame.data import Dataset
from torch_geometric.typing import NodeType
import pandas as pd
import numpy as np
from ast import literal_eval as make_tuple

PSEUDO_TIME = "pseudo_time"
TRAIN_SET_TIMESTAMP = pd.Timestamp("1970-01-01")
SRC_ENTITY_TABLE = "user_table"
DST_ENTITY_TABLE = "item_table"
TRANSACTION_TABLE = "transaction_table"
SRC_ENTITY_COL = "user_id"
DST_ENTITY_COL = "item_id"


class ContextGNN(RecMixin, BaseRecommenderModel):
    r"""
    ContextGNN: Beyond Two-Tower Recommendation Systems (https://arxiv.org/abs/2411.19513)
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_n_layers", "n_layers", "n_layers", 1, int, None),
            ("_norm", "norm", "norm", 'layer_norm', str, None),
            ("_sup_rat", "sup_rat", "sup_rat", 0.5, float, None),
            ("_channels", "channels", "channels", 128, int, None),
            ("_max_steps", "max_steps", "max_steps", 2000, int, None),
            ("_aggr", "aggr", "aggr", 'sum', str, None),
            ("_neigh", "neigh", "neigh", "(16,16,16,16)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
        ]
        self.autoset_params()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed_everything(self._seed)
        torch.use_deterministic_algorithms(True)

        if self._batch_size < 1:
            self._batch_size = self._num_users

        src_df, dst_df, target_df, train_df, test_df = data.get_relbench_df()
        del data
        NUM_SRC_NODES = len(src_df)
        self.NUM_DST_NODES = len(dst_df)

        table_dict = {
            SRC_ENTITY_TABLE: src_df,
            DST_ENTITY_TABLE: dst_df,
            TRANSACTION_TABLE: target_df,
        }

        def get_static_stype_proposal(
                table_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, stype]]:
            r"""Infer style for table columns."""
            inferred_col_to_stype_dict = {}
            for table_name, df in table_dict.items():
                df = df.sample(min(1_000, len(df)))
                inferred_col_to_stype = infer_df_stype(df)
                inferred_col_to_stype_dict[table_name] = inferred_col_to_stype
            return inferred_col_to_stype_dict

        col_to_stype_dict = get_static_stype_proposal(table_dict)

        def static_data_make_pkey_fkey_graph(
                table_dict: Dict[str, pd.DataFrame],
                col_to_stype_dict: Dict[str, Dict[str, stype]],
        ) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
            data = HeteroData()
            col_stats_dict = dict()
            # Update src nodes information in HeteroData and col_stats_dict
            src_col_to_stype = {"__const__": stype.numerical}
            src_df = pd.DataFrame(
                {"__const__": np.ones(len(table_dict[SRC_ENTITY_TABLE]))})
            src_dataset = Dataset(
                df=src_df,
                col_to_stype=src_col_to_stype,
            ).materialize()
            data[SRC_ENTITY_TABLE].tf = src_dataset.tensor_frame
            data[SRC_ENTITY_TABLE].time = torch.from_numpy(
                to_unix_time(table_dict[SRC_ENTITY_TABLE][PSEUDO_TIME]))
            col_stats_dict[SRC_ENTITY_TABLE] = src_dataset.col_stats
            # TODO: Remove the id features and add constant features somewhere

            # Update dst nodes information in HeteroData and col_stats_dict
            dst_col_to_stype = {"__const__": stype.numerical}
            dst_df = pd.DataFrame(
                {"__const__": np.ones(len(table_dict[DST_ENTITY_TABLE]))})
            dst_dataset = Dataset(
                df=dst_df,
                col_to_stype=dst_col_to_stype,
            ).materialize()
            data[DST_ENTITY_TABLE].tf = dst_dataset.tensor_frame
            data[DST_ENTITY_TABLE].time = torch.from_numpy(
                to_unix_time(table_dict[DST_ENTITY_TABLE][PSEUDO_TIME]))
            col_stats_dict[DST_ENTITY_TABLE] = dst_dataset.col_stats

            fkey_index = torch.from_numpy(
                table_dict[TRANSACTION_TABLE][SRC_ENTITY_COL].astype(int).values)
            pkey_index = torch.from_numpy(
                table_dict[TRANSACTION_TABLE][DST_ENTITY_COL].astype(int).values)
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (SRC_ENTITY_TABLE, SRC_ENTITY_COL, DST_ENTITY_TABLE)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            reverse_edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            reverse_edge_type = (DST_ENTITY_TABLE, DST_ENTITY_COL, SRC_ENTITY_TABLE)
            data[reverse_edge_type].edge_index = sort_edge_index(reverse_edge_index)
            data.validate()
            return data, col_stats_dict

        data, col_stats_dict = static_data_make_pkey_fkey_graph(
            table_dict=table_dict,
            col_to_stype_dict=col_to_stype_dict,
        )

        # num_neighbors = [
        #     int(args.num_neighbors // 2**i) for i in range(args.num_layers)
        # ]
        # num_neighbors = [16, 8, 8, 4]
        # num_neighbors = [8, 8, 8, 8]
        num_neighbors = self._neigh

        def static_get_link_train_table_input(
                transaction_df: pd.DataFrame,
                num_dst_nodes: int,
        ) -> (LinkTrainTableInput, Tensor):
            df = transaction_df
            src_node_idx: Tensor = torch.from_numpy(
                df[SRC_ENTITY_COL].astype(int).values)
            exploded = df[DST_ENTITY_COL].explode().dropna()

            coo_indices = torch.from_numpy(
                np.stack([exploded.index.values,
                          exploded.values.astype(int)]))
            sparse_coo = torch.sparse_coo_tensor(
                coo_indices,
                torch.ones(coo_indices.size(1), dtype=bool),  # type: ignore
                (len(src_node_idx), num_dst_nodes),
            )
            dst_node_indices = sparse_coo.to_sparse_csr()
            time = torch.from_numpy(to_unix_time(df[PSEUDO_TIME]))
            return LinkTrainTableInput(
                src_nodes=(SRC_ENTITY_TABLE, src_node_idx),
                dst_nodes=(DST_ENTITY_TABLE, dst_node_indices),
                num_dst_nodes=num_dst_nodes,
                src_time=time,
            )

        self.loader_dict: Dict[str, NeighborLoader] = {}
        self.dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
        num_dst_nodes_dict: Dict[str, int] = {}
        num_src_nodes_dict: Dict[str, int] = {}
        for split in ["train", "test"]:
            if split == "train":
                table = train_df
            elif split == "test":
                table = test_df
            table_input = static_get_link_train_table_input(table, num_dst_nodes=self.NUM_DST_NODES)
            self.dst_nodes_dict[split] = table_input.dst_nodes

            num_src_nodes_dict[split] = NUM_SRC_NODES
            num_dst_nodes_dict[split] = table_input.num_dst_nodes

            self.loader_dict[split] = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                time_attr="time",
                input_nodes=table_input.src_nodes,
                input_time=table_input.src_time,
                subgraph_type="bidirectional",
                batch_size=self._batch_size,
                shuffle=split == "train",
                num_workers=0,
                persistent_workers=False,
                disjoint=True,
            )

        self._model = ContextGNNModel(
            data=data,
            col_stats_dict=col_stats_dict,
            rhs_emb_mode=RHSEmbeddingMode.FUSION,
            dst_entity_table=DST_ENTITY_TABLE,
            num_nodes=num_dst_nodes_dict["train"],
            num_layers=self._n_layers,
            channels=self._channels,
            aggr=self._aggr,
            norm=self._norm,
            lr=self._learning_rate,
            embedding_dim=self._factors,
            torch_frame_model_kwargs={
                "channels": self._channels,
                "num_layers": self._n_layers,
            },
            is_static=True,
            src_entity_table=SRC_ENTITY_TABLE,
            num_src_nodes=NUM_SRC_NODES,
        )

    @property
    def name(self):
        return "ContextGNN" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        process = psutil.Process()
        for it in self.iterate(self._epochs):
            self._model.train()
            loss_accum = count_accum = 0
            steps = 0
            total_steps = min(len(self.loader_dict["train"]), self._max_steps)
            sparse_tensor = SparseTensor(self.dst_nodes_dict["train"][1], device=self.device)
            with tqdm(total=total_steps, disable=not self._verbose) as t:
                for batch in self.loader_dict["train"]:
                    batch = batch.to(self.device)

                    # Get ground truth edges
                    input_id = batch[SRC_ENTITY_TABLE].input_id
                    src_batch, dst_index = sparse_tensor[input_id]
                    edge_label_index = torch.stack([src_batch, dst_index], dim=0)

                    train_seed_nodes = self.loader_dict["train"].input_nodes[1].to(
                        src_batch.device)
                    global_src_index = train_seed_nodes[
                        batch[SRC_ENTITY_TABLE].input_id[src_batch]]
                    global_edge_label_index = torch.stack([global_src_index, dst_index],
                                                          dim=0)
                    supervision_edgse_sample_size = int(global_edge_label_index.shape[1] *
                                                        self._sup_rat)
                    sample_indices = torch.randperm(global_edge_label_index.shape[1],
                                                    device=global_edge_label_index.device
                                                    )[:supervision_edgse_sample_size]
                    global_edge_label_index_sample = (
                        global_edge_label_index[:, sample_indices])
                    # Update edge_label_index to match.
                    edge_label_index = edge_label_index[:, sample_indices]

                    # batch.edge_type=[
                    # ('user_table', 'user_id', 'item_table'),
                    # ('item_table', 'item_id', 'user_table'),
                    # ]
                    edge_type = (SRC_ENTITY_TABLE, SRC_ENTITY_COL, DST_ENTITY_TABLE)
                    edge_index = batch[edge_type].edge_index

                    # NOTE: Assume that dst node indices are consecutive
                    # starting from 0 and monotonically increasing, which is
                    # true for all 3 static datasets: amazon-book, gowalla and
                    # yelp2018 that we have.
                    global_src_index = batch[SRC_ENTITY_TABLE].n_id[edge_index[0]]
                    global_dst_index = batch[DST_ENTITY_TABLE].n_id[edge_index[1]]
                    global_edge_index = torch.stack([global_src_index, global_dst_index])

                    # Create a mask to track the supervision edges for each disjoint
                    # subgraph in a batch
                    global_src_batch = batch[SRC_ENTITY_TABLE].batch[edge_index[0]]
                    # global_dst_batch = batch[DST_ENTITY_TABLE].batch[edge_index[1]]
                    # NOTE: assert all(global_src_batch == global_dst_batch) is True
                    global_seed_nodes = train_seed_nodes[
                        batch[SRC_ENTITY_TABLE].input_id[global_src_batch]]
                    supervision_seed_node_mask = (
                            global_seed_nodes == global_edge_index[0])

                    global_edge_label_index_hash = global_edge_label_index_sample[
                                                   0, :] * self.NUM_DST_NODES + global_edge_label_index_sample[1, :]
                    global_edge_index_hash = global_edge_index[
                                             0, :] * self.NUM_DST_NODES + global_edge_index[1, :]

                    # Mask to filter out edges in edge_index_hash that are in
                    # edge_label_index_hash
                    mask = ~(
                            torch.isin(global_edge_index_hash, global_edge_label_index_hash) *
                            supervision_seed_node_mask)

                    # TODO (xinwei): manually swtich the direct and reverse edges
                    # Apply the mask to filter out the ground truth edges
                    edge_index_message_passing = edge_index[:, mask]
                    # unique_nodes_row = torch.unique(edge_index_message_passing[0])
                    # unique_nodes_col = torch.unique(edge_index_message_passing[1])
                    # num_unique_nodes_row = len(unique_nodes_row)
                    # num_unique_nodes_col = len(unique_nodes_col)
                    edge_index_message_passing_sparse = torch.sparse_coo_tensor(edge_index_message_passing.to(self.device),
                                                                                torch.ones(edge_index_message_passing.shape[1]).to(self.device))
                    edge_index_message_passing_reverse_sparse = torch.sparse_coo_tensor(edge_index_message_passing.flip(dims=[0]).to(self.device),
                                                                                        torch.ones(edge_index_message_passing.shape[1]).to(self.device))
                    # batch[edge_type].edge_index = edge_index_message_passing_sparse
                    batch[edge_type].edge_index = edge_index_message_passing_reverse_sparse
                    reverse_edge_type = (DST_ENTITY_TABLE, DST_ENTITY_COL,
                                         SRC_ENTITY_TABLE)
                    # batch[reverse_edge_type].edge_index = edge_index_message_passing
                    batch[reverse_edge_type].edge_index = edge_index_message_passing_sparse

                    # Optimization
                    self._model.optimizer.zero_grad()
                    logits = self._model(batch, SRC_ENTITY_TABLE, DST_ENTITY_TABLE)

                    loss = sparse_cross_entropy(logits, edge_label_index)

                    numel = len(batch[DST_ENTITY_TABLE].batch)

                    loss.backward()
                    self._model.optimizer.step()

                    loss_accum += float(loss) * numel
                    count_accum += numel
                    steps += 1

                    if steps > self._max_steps:
                        break

                    mem_usage = process.memory_info().rss / (1024 ** 3)
                    gpu_usage = torch.cuda.max_memory_reserved(self.device) / 1024 ** 3
                    t.set_postfix(
                        {'loss': f'{loss / steps:.5f}', 'cpu': f'{mem_usage:.2f}GB', 'gpu': f'{gpu_usage:.2f}GB'})
                    t.update()

            if count_accum == 0:
                print(f"Did not sample a single '{DST_ENTITY_TABLE}' "
                      f"node in any mini-batch. Try to increase the number "
                      f"of layers/hops and re-try. If you run into memory "
                      f"issues with deeper nets, decrease the batch size.")
            epoch_loss = loss_accum / count_accum if count_accum > 0 else float("nan")
            t.set_postfix({'epoch loss': f'{epoch_loss:.5f}'})
            t.update()

            self.evaluate(it, epoch_loss)

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}

        self._model.eval()
        for batch in self.loader_dict["test"]:
            batch = batch.to(self.device)
            batch_size = batch[SRC_ENTITY_TABLE].batch_size

            out = self._model(batch, SRC_ENTITY_TABLE, DST_ENTITY_TABLE).detach()
            scores = torch.sigmoid(out)

            # Map local batch indices to global indices
            global_batch_src_ids = batch[SRC_ENTITY_TABLE].n_id[:batch_size]

            recs_val, recs_test = self.process_protocol(k, scores, global_batch_src_ids)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def process_protocol(self, k, *args):

        if not self._negative_sampling:
            recs = self.get_single_recommendation(self.get_candidate_mask(), k, *args)
            return recs, recs
        else:
            return self.get_single_recommendation(self.get_candidate_mask(validation=True), k, *args) if hasattr(
                self._data, "val_dict") else {}, \
                self.get_single_recommendation(self.get_candidate_mask(), k, *args)

    def get_single_recommendation(self, mask, k, predictions, global_batch_src_ids):
        v, i = self._model.get_top_k(predictions, mask[global_batch_src_ids.detach().cpu().numpy()], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(list(itemgetter(*global_batch_src_ids.cpu().numpy().tolist())(self._data.private_users)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
