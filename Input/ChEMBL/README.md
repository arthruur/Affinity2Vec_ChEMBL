# How we built the dataset
We downloaded ChEMBL v34 database and filtered 47,506 activities of ChEMBLv34 using the following criteria:
- assay_type = B (for binding)
- target_type = Single protein
- assay_organism = Homo sapiens
- pchembl_value NOT NULL
- max_phase NOT NULL

Then we took the average of pchembl_value for the repeated pairs, getting 16,531 unique (drug,target) pairs. For training, we selected only the pairs in which the compounds have SMILES greater than 5 characters (total of 3,340 compounds) and the targets that are not between the 31 Repo3D proteins (total of 1526 no Repo3D targets).

We got all SMILES from the [PubChem Identifier Exchange Service](https://pubchem.ncbi.nlm.nih.gov/docs/identifier-exchange-service) by inputting a list of ChEMBL IDs of the compounds. We extracted from the set all compounds that do not have structures stored in PubChem.

We got all FASTA for the targets from [Uniprot](https://www.uniprot.org/).

## Description of the dataset files:
- *Input/ChEMBL/compoundsChEMBL34_SMILESgreaterThan5.tsv* = compounds with maxPhase record not null and length os SMILES greater than 5.
- *Input/ChEMBL/targetsChEMBL34_noRepo3D.tsv* = targets of Human proteome without the Repo3D proteins
- *Input/ChEMBL/affinitiesChEMBL34.tsv* = (drug,target) affinities using pChembl. A high pChembl means a high affinity.


# How we built the input files

## Compound similarity matrix
The compound similarity matrix (DDsim1 matrix) should be calculated using the SIMCOMP tool (https://pubs.acs.org/doi/10.1021/ja036030u) which represents the compounds by their two-dimensional (2D) chemical structure (as a graph). Then, the compound-compound similarity score is based on the common sub-structure size of the two graphs using the Tanimoto coefficient. One must use the SIMCOMP2 tool available at https://www.genome.jp/tools/simcomp2 to compare compounds' similarity. 

The SIMCOMP tool requires the compounds to be identified by a KEGG ID or MOL text file. To convert PubChem CID to KEGG we tested the MataboAnalyst ConvertView Web Service (https://www.metaboanalyst.ca/MetaboAnalyst/upload/ConvertView.xhtml). Unfortunately, not all compounds have a KEGG ID. To convert SMILES to MOL would take a lot of time. Thus, for initial experiments, we decided to compute the compound-compound similarity using the Tanimoto coefficient from the RDKit library (see the script [../../ang_compound2compound-similarity.py](../../ang_compound2compound-similarity.py)).

The authors suggest to filter the compounds with similarity score lower than a certain threshold manually established. To evaluate the similarity score to define the tresholds run the script (ang_plotSimilarities.py)[../../ang_plotSimilarities.py].


## Target similarity matrix
The target similarity matrix (TTsim1 matrix) was calculated using the normalized Smith-Waterman (SW) scores (https://doi.org/10.1093/bioinformatics/btn162) based on the protein sequences’ alignment. To compute the SW scores we used the [py_stringmatching](https://pypi.org/project/py-stringmatching/) package from [AnHai Doan's group at the University of Wisconsin-Madison](https://sites.google.com/site/anhaidgroup/current-projects/magellan/py_stringmatching).

The normalized simlarity score was computed as described in [Prediction of compound–target interaction networks from the integration of chemical and genomic spaces](https://academic.oup.com/bioinformatics/article/24/13/i232/231871): SWN(g,g')=SW(g,g')/(sqrt(SW(g,g))*sqrt(SW(g',g'))), where SW(·,·) means the Smith–Waterman score, and g and g' are two different proteins. We built the similarity score matrix by applying this operation to all protein pairs. The script to compute the matrix is [ang_target2target-SWNormalizedSimilarity.py](../../ang_target2target-SWNormalizedSimilarity.py) by using the scores previously computed in [ang_target2target-SWsimilarity.py](../../ang_target2target-SWsimilarity.py).

The authors suggest to filter the targets with similarity score lower than a certain threshold manually established. See  [Filtering similarity edge](#filtering-similarity-edges).


## Compound embeddings
Authors didn't explain how to compute the embeddings using the model included in the repository. We had to adapt the original seq2seq fingerprint code to generate the drug embedings. Our code is at https://github.com/angeload/seq2seq-fingerprint-repo3d. The files generated from ChEMBL are to big to include in the GitHub repository. So we left them in a cloud drive.

## Target embeddings
The authors adopted the Prot2Vec model available at https://github.com/ehsanasgari/Deep-Proteomics. The raw protein ngrams are also available at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN. Even though the authors included the model for generate protein's embeddings [swissprot-reviewed-protvec.model](../Trained_models/swissprot-reviewed-protvec.model), they don't explain how to use it. The instructions about how to use the model to grenerate embeddings are in the BIOVEC libray GitHUb repository available at https://github.com/kyu999/biovec.

*!!!ATTENTION!!! The original BIOVEC library rises the error "ImportError: cannot import name 'Fasta' from 'fasta'". To solve it just changed the line 'from pyfasta import Fasta' to 'from pyfaidx import Fasta' in the utils.py script in the library (see issue #15 in the GitHub repository).*

# Preprocessing and constructing DTBA network.

## Filtering similarity edges.
 In the tranning script, drug-drug similarity matrix (DDsim) and target-target similarity matrix (TTsim) are filtered so nodes with similarity bellow a certain score were removed. To determine the score for each matrix use the [ang_plotSimilarities.py](ang_plotSimilarities.py) script to plot the cumulative curve of similarities and visualy define a threshold so similarities bellow a certain value are discarded (see how authors made this in the suplementary information of the paper).

 
