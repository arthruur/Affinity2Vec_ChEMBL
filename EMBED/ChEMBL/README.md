# How we created the embeddings

## Drug embeddings
Authors of Affinty2Vec offered a trained model for drug embeddings based on Seq2Seq model [trained_seq2seq_SMILES_model.pickle.dat](../../Trained_models/trained_seq2seq_SMILES_model.pickle.dat), but have not explained how to use it.

See [seq2seq-fingerprint-repo3d](https://github.com/angeload/seq2seq-fingerprint-repo3d) repository on GitHub to a detailed explanation about trainning a model to get the fingerprints of the drugs.

 We used the drugs listed in [../../Input/compoundsChEMBL34_SMILESgreaterThan5.tsv](../../Input/compoundsChEMBL34_SMILESgreaterThan5.tsv). This file has all SMILES with more than 5 tokens selected from ChEMBL for compounds associated with Homo sapiens. 

## Target embeddings
Authors used the [BIOVEC package]( https://github.com/kyu999/biovec) to compute the embeddings. The BIOVEC repository explain how we can use the trained model that the authors used [swissprot-reviewed-protvec.model](../../Trained_models/swissprot-reviewed-protvec.model).

We used the targets listed in [targetsChEMBL34_noRepo3D.tsv](../../Input/targetsChEMBL34_noRepo3D.tsv) and created the embeddings using [ang_targetEmbeddings.py](../../ang_targetEmbeddings.py). We do not use the 31 proteins from the Repo3D list for they were used for test the model.
 