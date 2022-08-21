tcpentlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol TCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 2 \
			--do_train                                                      \
			--warmup\
			--handmaded_label_map_path data/label_map_handmaded_02.json
#			--label_map_path data/label_map_data_ratio0.6_multitoken_top6.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json  \


dccpentlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol DCCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 3 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
#			--label_map_path data/label_map_data_ratio0.6_multitoken_top6.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json

bgpv4entlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol BGPv4                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 1 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
#			--label_map_path data/label_map_handmaded_multitoken_top6.json

ltpentlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol LTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 2 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
			-#-label_map_path data/label_map_handmaded_multitoken_top6.json

pptpentlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol PPTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 1 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
			-#-label_map_path data/label_map_handmaded_multitoken_top6.json

sctpentlm:
	python3 models/entlm_rfc.py                        			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--protocol SCTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 1e-5                                            \
			--batch_size 1 	\
			--patience 10 \
			--cuda_device 0 \
			--do_train                                                      \
			--warmup \
			--handmaded_label_map_path data/label_map_handmaded_02.json
			-#-label_map_path data/label_map_handmaded_multitoken_top6.json

tcpberttrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol TCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 2 \
			--template_num 3 \
		  	--template_id 2 \
		  	--multi_template \
		  	--few_shot 5
#			--do_train                                                      \
#		  	--soft_template\
#		  	--template_len 20

tcpfewshot:
	python3 models/bert_bert_mlp_cuda_few-shot.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol TCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 1 \
			--template_num 3 \
		  	--template_id 2 \
		  	--multi_template

dccpfewshot:
	python3 models/bert_bert_mlp_cuda_few-shot.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol DCCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 1 \
			--template_num 3 \
		  	--template_id 0 \
#		  	--multi_template

dccpberttrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol DCCP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 3 \
			--template_num 3 \
		  	--template_id 2 \
		  	--multi_template

bgpv4berttrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol BGPv4                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 1 \
			--template_num 3 \
		  	--template_id 0

ltpberttrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol LTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 3 \
			--template_num 3 \
		  	--template_id 0

pptpberttrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol PPTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 2 \
			--template_num 3 \
		  	--template_id 0

sctpberttrain:
	python3 models/bert_bert_mlp_cuda.py                          			\
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol SCTP                                                  \
			--outdir output 												\
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 	\
			--cuda_device 3 \
			--template_num 3 \
		  	--template_id 0
# --------- original rfcnlp --------------
obgpv4berttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol BGPv4                                                  \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0

opptpberttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol PPTP                                                  \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0

osctpberttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol SCTP                                                  \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0

oltpberttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol LTP                                                  \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0

odccpberttrain:
	python3 models/bert_bilstm_crf.py                           \
    		--features                                                      \
    		--savedir .                                                     \
    		--do_train                                                      \
    		--do_eval                                                       \
    		--heuristics                                                    \
    		--protocol DCCP                                                 \
    		--outdir output                                                 \
    		--bert_model networking_bert_rfcs_only                          \
    		--learning_rate 2e-5                                            \
    		--batch_size 1                                                  \
    		--cuda_device 0


# ------- linear -------

dccplineartrain:
	python3 models/linear.py            \
		--protocol DCCP                         \
		--stem                                  \
		--phrase_level                          \
		--outdir output                         \

tcplineartrain:
	python3 models/linear.py            \
		--protocol TCP                          \
		--stem                                  \
		--phrase_level                          \
		--outdir output                         \


# ------- cpu --------
tcpberttraincpu:
	python3 models/bert_bert_mlp.py                          \
			--features                                                      \
			--savedir .                                                     \
			--do_train                                                      \
			--do_eval                                                       \
			--heuristics                                                    \
			--protocol TCP                                                 \
			--outdir output \
			--bert_model networking_bert_rfcs_only                          \
			--learning_rate 2e-5                                            \
			--batch_size 1 \

