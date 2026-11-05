Tisztelt iskolatársak! You will need to specify a path to the directory where the dataset is. Create a .env file, and inside there should be one environment variable DATASET_PATH, for instance DATASET_PATH='./Dataset/ParlaMint-HU/ParlaMint-HU.conllu' . If you create a directory Dataset in the project, it is in the .gitignore so will not be committed to git. This way the dataset itself does not need to be committed to git, since it is very large and will make the repo difficult to clone.

Citations:

Erjavec, Tomaž; et al., 2025, 
  Linguistically annotated multilingual comparable corpora of parliamentary debates ParlaMint.ana 5.0, Slovenian language resource repository CLARIN.SI, ISSN 2820-4042, 
  http://hdl.handle.net/11356/2005.
