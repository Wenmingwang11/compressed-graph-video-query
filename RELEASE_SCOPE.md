# Submission-stage release scope

The associated manuscript is being submitted to EDBT; it has not been accepted. Core implementation is temporarily withheld.

## Withheld modules

- `apriori.py`
- `apriori_pro.py`
- `batch_query.py`
- `cp_graph.py`
- `Fp_growth.py`
- `FPgrowth.py`
- `lists_cluster.py`
- `paper_query.py`
- `paper_query_corrected.py`
- `paper_query_pairs.py`
- `prefix_tree.py`
- `query.py`
- `subsets.py`
- `supplement/paper_experiment_utils.py`
- `tests/test_corrected_mli_querying.py`
- `tests/test_corrected_querying.py`
- `tests/test_mul_build_index.py`
- `tests/test_paper_experiment_utils.py`
- `tests/test_paper_query_corrected.py`
- `tests/test_paper_querying.py`
- `tests/test_paper_querying_pairs.py`
- `tests/test_prepared_mli_querying.py`
- `tests/test_prepared_mli_querying_fast.py`
- `Tree_Node.py`
- `vsimsearch/build_index.py`
- `vsimsearch/build_index_procssing.py`
- `vsimsearch/corrected_mli_querying.py`
- `vsimsearch/corrected_querying.py`
- `vsimsearch/indexing.py`
- `vsimsearch/mul_build_index.py`
- `vsimsearch/paper_querying.py`
- `vsimsearch/paper_querying_pairs.py`
- `vsimsearch/prepared_mli_querying.py`
- `vsimsearch/prepared_mli_querying_fast.py`
- `vsimsearch/querying.py`

These files are explicit placeholders, not working implementations. They raise NotImplementedError. Related tests are withheld where they reveal internal implementation. Public code cannot execute the complete indexing and query pipeline until these modules are released.

## Included materials

Main project Python sources, baseline integration code, supplementary experiments, available tests, dependency specification, and manuscript figures. These are research scripts; some retain original local dataset paths and require path configuration and separately obtained dependencies/data. Existing adapter variants are not claims of complete reproduction of every original baseline.

## Excluded assets

Raw videos, object records, manual annotations, serialized indexes, model weights, environments, credentials, caches, editor state, internal logs and experiment output directories. Third-party model/system repositories are not bundled. No new license is granted for third-party code; upstream ownership and applicable licensing remain unchanged.
