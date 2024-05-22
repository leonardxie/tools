import weakref
from typing import List, Optional, Tuple

import torch

from vllm.sequence import ExecuteModelRequest, SamplerOutput, SequenceData
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker_base import LoraNotSupportedWorkerBase

import transformers
from transformers import AutoTokenizer

class NGramWorker(LoraNotSupportedWorkerBase):
    """NGramWorker provides a light drafter without need for model.

    Current NGramWorker only implement prompt lookup decoding,
    and in future we may also do RAG type drafter and other scenerios
    which don't rely on LLM model to give proposals.
    """

    def __init__(self, *args, **kwargs):
        # Get local_rank/vocab_size from kwargs attribute
        self.local_rank = kwargs["local_rank"]
        self.vocab_size = kwargs["model_config"].get_vocab_size()

        # Lazy initialization list.
        self._proposer: Top1Proposer

        # 自己加入的新的参数
        self.ngram_probabilities = {}
        self.N = 4
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs["model_config"].tokenizer, trust_remote_code=True)

    def set_ngram_window_size(self, ngram_prompt_lookup_min: int,
                              ngram_prompt_lookup_max: int):
        # Search valid candidate window between
        # ngram_prompt_lookup_min/ngram_prompt_lookup_max
        self.ngram_prompt_lookup_max = ngram_prompt_lookup_max
        self.ngram_prompt_lookup_min = ngram_prompt_lookup_min

    def init_device(self):
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.load_model = lambda *args, **kwargs: None

        # Current only support Top1Proposer
        self._proposer = Top1Proposer(
            weakref.proxy(self),
            device=self.device,
            vocab_size=self.vocab_size,
        )

    def set_include_gpu_probs_tensor(self):
        # NGram don't need gpu sampler
        pass

    def execute_model(self, execute_model_req: ExecuteModelRequest) -> None:
        """NGram doesn't depend on model execution, just pass this function"""
        # 此处， NGram模型计算概率，并保存成dict，用于decoding阶段的计算
        if execute_model_req: # 处理多卡模型的bug
            for seq_group_meta_data in execute_model_req.seq_group_metadata_list:
                seq_request_id = seq_group_meta_data.request_id
                if seq_request_id in self.ngram_probabilities.keys():
                    continue
                else:
                    probability_list = self.compute_ngram_probabilities(
                        text=seq_group_meta_data.sampling_params.ngram_text
                    )
                    self.ngram_probabilities[seq_request_id] = probability_list

    def compute_ngram_probabilities(self, text):
        content_ids = self.tokenizer.encode(text)
        keys4, dicWordFrequency, dicPhraseFrequency1, dicPhraseFrequency2, dicPhraseFrequency3 = self.count(content_ids)
        probability_w1w4, probability_w2w4, probability_w3w4, probability_w4 = self.compute_probabilities(keys4, dicWordFrequency, dicPhraseFrequency1, dicPhraseFrequency2, dicPhraseFrequency3, content_ids)
        return [probability_w1w4, probability_w2w4, probability_w3w4, probability_w4]

    def get_text_iterator(self, txt):
        '''
        text模型迭代器
        :param txt: 一段话或一个句子
        :return: 返回迭代器，item 为 tuple，每项 2 个值
        '''
        ct = len(txt)
        if ct < self.N:
            return txt
        for i in range(ct - self.N + 1):
            yield txt[i:i+self.N]

    def count(self, content_ids):
        '''
        训练 ngram  模型
        :param content: 训练内容
        :return:
        '''
        dicWordFrequency = dict()  # 临时词频
        dicPhraseFrequency1 = dict()  # 临时词段频
        dicPhraseFrequency2 = dict()  # 临时词段频
        dicPhraseFrequency3 = dict()  # 临时词段频
        dicPhraseFrequency4 = dict()  # 临时词段频  # 临时词段概率
        count = 1
        ie = self.get_text_iterator(content_ids)  # bi-Gram 模型
        keys1, keys2, keys3, keys4 = [], [], [], []
        for w in ie:
            # 词频
            k1, k2, k3, k4 = w[0], w[1], w[2], w[3]

            if k1 not in dicWordFrequency.keys():
                dicWordFrequency[k1] = 0
            if k2 not in dicWordFrequency.keys():
                dicWordFrequency[k2] = 0
            if k3 not in dicWordFrequency.keys():
                dicWordFrequency[k3] = 0
            if k4 not in dicWordFrequency.keys():
                dicWordFrequency[k4] = 0

            if count == 1:
                dicWordFrequency[k1] += 4
                dicWordFrequency[k2] += 3
                dicWordFrequency[k3] += 2
                dicWordFrequency[k4] += 1
            elif count == len(content_ids) - 3:
                dicWordFrequency[k1] += 1
                dicWordFrequency[k2] += 2
                dicWordFrequency[k3] += 3
                dicWordFrequency[k4] += 4
            else:
                dicWordFrequency[k1] += 1
                dicWordFrequency[k2] += 1
                dicWordFrequency[k3] += 1
                dicWordFrequency[k4] += 1

            # 词段频
            key1 = (k1, k4)
            key2 = (k2, k4)
            key3 = (k3, k4)
            key4 = (k1, k2, k3, k4)
            keys1.append(key1)
            if count == 1:
                keys2.append((k1, k3))
                if (k1, k3) not in dicPhraseFrequency2.keys():
                    dicPhraseFrequency2[(k1, k3)] = 0
                dicPhraseFrequency2[(k1, k3)] += 1

                keys3.append((k1, k2))
                keys3.append((k2, k3))
                if (k1, k2) not in dicPhraseFrequency3.keys():
                    dicPhraseFrequency3[(k1, k2)] = 0
                dicPhraseFrequency3[(k1, k2)] += 1
                if (k2, k3) not in dicPhraseFrequency3.keys():
                    dicPhraseFrequency3[(k2, k3)] = 0
                dicPhraseFrequency3[(k2, k3)] += 1

            keys2.append(key2)
            keys3.append(key3)
            keys4.append(key4)
            if key1 not in dicPhraseFrequency1.keys():
                dicPhraseFrequency1[key1] = 0
            dicPhraseFrequency1[key1] += 1
            if key2 not in dicPhraseFrequency2.keys():
                dicPhraseFrequency2[key2] = 0
            dicPhraseFrequency2[key2] += 1
            if key3 not in dicPhraseFrequency3.keys():
                dicPhraseFrequency3[key3] = 0
            dicPhraseFrequency3[key3] += 1

            if key4 not in dicPhraseFrequency4.keys():
                dicPhraseFrequency4[key4] = 0
            dicPhraseFrequency4[key4] += 1
            count += 1
        # 修改词概率
        dicWordFrequency = {key: value/4 for key, value in dicWordFrequency.items()}

        return keys4, dicWordFrequency, dicPhraseFrequency1, dicPhraseFrequency2, dicPhraseFrequency3

    def compute_probabilities(self, keys4, dicWordFrequency, dicPhraseFrequency1,
                              dicPhraseFrequency2, dicPhraseFrequency3, content_ids,):
        # 词段概率
        Probability_w1w4, Probability_w2w4, Probability_w3w4, Probability_w4 = dict(), dict(), dict(), dict()
        for w1w2w3w4 in keys4:
            w1, w2, w3, w4 = w1w2w3w4
            w1w4 = (w1w2w3w4[0], w1w2w3w4[3])
            w2w4 = (w1w2w3w4[1], w1w2w3w4[3])
            w3w4 = (w1w2w3w4[2], w1w2w3w4[3])
            w1Freq, w2Freq, w3Freq, w4Freq = dicWordFrequency[w1], dicWordFrequency[w2], dicWordFrequency[w3], \
                                             dicWordFrequency[w4]

            w1w4Freq = dicPhraseFrequency1[w1w4]
            w2w4Freq = dicPhraseFrequency2[w2w4]
            w3w4Freq = dicPhraseFrequency3[w3w4]

            # 独立概率分布的计算概率
            Probability_w1w4[w1w4] = round((w1w4Freq / w4Freq), 4)
            Probability_w2w4[w2w4] = round((w2w4Freq / w4Freq), 4)
            Probability_w3w4[w3w4] = round((w3w4Freq / w4Freq), 4)
            Probability_w4[w4] = round((w4Freq / len(content_ids)), 4)

        # Probability_w3w4[(151645, 151643)] = 999
        device = self.device
        shape = self.vocab_size
        # 稀疏矩阵添加数据
        tensor_keys_w1w4 = torch.tensor([list(key) for key in list(Probability_w1w4.keys())]).to(device)
        tensor_values_w1w4 = torch.tensor(list(Probability_w1w4.values())).to(device)
        probability_w1w4 = torch.sparse_coo_tensor(
            torch.stack([tensor_keys_w1w4[:, 0], tensor_keys_w1w4[:, 1]]),
            tensor_values_w1w4,
            torch.Size((shape, shape))).coalesce().to(device)

        tensor_keys_w2w4 = torch.tensor([list(key) for key in list(Probability_w2w4.keys())]).to(device)
        tensor_values_w2w4 = torch.tensor(list(Probability_w2w4.values())).to(device)
        probability_w2w4 = torch.sparse_coo_tensor(
            torch.stack([tensor_keys_w2w4[:, 0], tensor_keys_w2w4[:, 1]]),
            tensor_values_w2w4,
            torch.Size((shape, shape))).coalesce().to(device)

        tensor_keys_w3w4 = torch.tensor([list(key) for key in list(Probability_w3w4.keys())]).to(device)
        tensor_values_w3w4 = torch.tensor(list(Probability_w3w4.values())).to(device)
        probability_w3w4 = torch.sparse_coo_tensor(
            torch.stack([tensor_keys_w3w4[:, 0], tensor_keys_w3w4[:, 1]]),
            tensor_values_w3w4,
            torch.Size((shape, shape))).coalesce().to(device)

        tensor_keys_w4 = torch.tensor([key for key in list(Probability_w4.keys())]).to(device)
        tensor_values_w4 = torch.tensor(list(Probability_w4.values())).to(device)
        probability_w4 = torch.sparse_coo_tensor(
            tensor_keys_w4.unsqueeze(0),
            tensor_values_w4,
            torch.Size([shape])).coalesce().to(device)

        return probability_w1w4, probability_w2w4, probability_w3w4, probability_w4


    def determine_num_available_blocks(self) -> None:
        """NGram doesn't depend on model execution, no need to check blocks"""
        pass

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """As there is no cache need to handle, just pass this function"""
        pass

    def get_cache_block_size_bytes(self):
        """Return the size of a cache block in bytes."""
        return 0

    def extract_match(
            self,
            seq_data: SequenceData,
            sample_len: int,
            token_id_list: List,
            token_prob_list: List,
            has_spec_out: bool,
    ):
        input_ids = torch.as_tensor(seq_data.get_token_ids(), dtype=torch.long, device=self.device)
        input_length = seq_data.get_len()

        for ngram_size in range(
                min(self.ngram_prompt_lookup_max, input_length - 1),
                self.ngram_prompt_lookup_min - 1,
                -1,
        ):
            ngram_tensor = input_ids[-ngram_size:]
            proposal_start_idx = None
            if ngram_size == 1:
                # Do not match itself and do not use unfold and all
                matches = (input_ids[:-1] == ngram_tensor)
            else:
                windows = input_ids.unfold(dimension=0,
                                           size=ngram_size,
                                           step=1)
                # Do not match itself
                matches = (windows[:-1] == ngram_tensor).all(dim=-1)

            # first_match includes "values" (bool), indicating whether
            # the match is found, and "indices", indicating the index
            # of the first match.
            # Note that "first_match.values.item()" triggers GPU-CPU
            # sync so it is a bit inefficient, but we have not found
            # a better way to do this.
            first_match = matches.max(dim=-1)
            if first_match.values.item():
                proposal_start_idx = first_match.indices.add_(ngram_size)
                spec_indices = (proposal_start_idx).repeat(sample_len) + torch.arange(sample_len, device=self.device)
                spec_indices.clamp_(max=input_ids.shape[-1] - 1)
                res = input_ids.gather(dim=-1, index=spec_indices)
                token_id_list.append(res)
                token_prob_list.append(torch.nn.functional.one_hot(res, num_classes=self.vocab_size).to(torch.float32))
                has_spec_out = True
                break
        else:
            token_id_list.append(None)
            token_prob_list.append(None)

        return token_id_list, token_prob_list, has_spec_out

    def bayes_ngram(
            self,
            seq_group_metadata,
            sample_len: int,
            token_id_list: List,
            token_prob_list: List,
            has_spec_out: bool,
    ):
        """
        n-gram概率计算
        """
        has_spec_out = True
        seq_request_id = seq_group_metadata.request_id
        seq_data = next(iter(seq_group_metadata.seq_data.values()))
        probability_list = self.ngram_probabilities[seq_request_id]
        token_ids = seq_data.get_token_ids()
        tokens_of_single_seq = []
        probs_of_single_seq = []

        for i in range(sample_len):
            input_ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
            p_w1w4 = probability_list[0][input_ids[-3]].to_dense()
            p_w2w4 = probability_list[1][input_ids[-2]].to_dense()
            p_w3w4 = probability_list[2][input_ids[-1]].to_dense()
            p_w4 = probability_list[3].to_dense()
            single_logits = torch.mul(p_w4, torch.mul(p_w3w4, torch.mul(p_w1w4, p_w2w4))).unsqueeze(0)
            if list(single_logits.nonzero()):
                greedy_samples = torch.argmax(single_logits, dim=-1)
                tokens_of_single_seq.append(int(greedy_samples))
                probs_of_single_seq.append(single_logits)
                token_ids.append(int(greedy_samples))
            else:
                has_spec_out = False
                break
        if has_spec_out:
            token_id_list.append(torch.as_tensor(tokens_of_single_seq, dtype=torch.long, device=self.device))
            token_prob_list.append(torch.cat(probs_of_single_seq, dim=0))
        else:
            token_id_list.append(None)
            token_prob_list.append(None)

        return token_id_list, token_prob_list, has_spec_out


    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[Optional[List[SamplerOutput]], bool]:
        """NGram match algo to pick proposal candidate. Returns the list of
        sampler output, one per SequenceGroupMetadata.

        For ngram worker, we already done needed transposed internal, so the
        indicator pass to sampler_output_to_torch shall be False.
        """
        self._raise_if_unsupported(execute_model_req)

        has_spec_out = False
        token_id_list = []
        token_prob_list = []
        for idx, seq_group_metadata in enumerate(
                execute_model_req.seq_group_metadata_list):
            # seq_data = next(iter(seq_group_metadata.seq_data.values()))

            # token_id_list, token_prob_list, has_spec_out = self.extract_match(
            #     seq_data, sample_len, token_id_list, token_prob_list, has_spec_out)

            token_id_list, token_prob_list, has_spec_out = self.bayes_ngram(seq_group_metadata,
                                                                            sample_len,
                                                                            token_id_list,
                                                                            token_prob_list,
                                                                            has_spec_out)

        if not has_spec_out:
            return None, False

        outputs: List[Optional[SamplerOutput]] = []
        for idx in range(len(execute_model_req.seq_group_metadata_list)):
            if token_id_list[idx] is None:
                outputs.append(None)
            else:
                outputs.append(
                    SamplerOutput(
                        outputs=None,
                        sampled_token_probs=token_prob_list[idx],
                        logprobs=torch.zeros((sample_len, self.vocab_size),
                                             dtype=torch.float32,
                                             device=self.device),
                        sampled_token_ids=token_id_list[idx],
                    ))

        return outputs, False

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """

        return self._proposer.get_proposals(execute_model_req)

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        """NGramWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "NGramWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "NGramWorker does not support beam search.")
