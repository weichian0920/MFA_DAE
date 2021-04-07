python ../src/main.py \
--model_type="DAE_C" \
--data_feature='lps' \
--optim='Adam' \
--batch_size=32 \
--lr=0.001 \
--epochs=100 \
--source_num=2 \
--clustering_alg=NMF \
--wienner_mask=True \
