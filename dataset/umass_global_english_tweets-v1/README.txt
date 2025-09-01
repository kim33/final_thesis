The UMass Global English on Twitter Dataset
Version 1, released September 8, 2017
Su Lin Blodgett, Johnny Tian-Zheng Wei, and Brendan O'Connor
http://slanglab.cs.umass.edu/TwitterLangID/

This dataset contains 10,502 tweets, randomly sampled from all publicly
available geotagged Twitter messages, annotated for being in English,
non-English, or having code switching, language ambiguity, or having been
automatically generated.  It includes messages sent from 130 different
countries.

The dataset is described in the paper:

@inproceedings{blodgett:2017wnut,
	Author = {Su Lin Blodgett and Johnny Tian-Zheng Wei and Brendan O'Connor},
	Booktitle = {Proceedings of the Workshop on Noisy User-Generated Text},
	Publisher = {Association for Computational Linguistics},
	Title = {Recognizing Global Social Media English with U.S. Demographic Modeling},
	Year = {2017},
}

If you make use of this dataset in research, please consider citing our paper.
Thanks!

The file all_annotated.tsv contains the dataset of 10,502 tweets used in the
paper.  Text is encoded as UTF-8.

The columnn headings (also given in the .tsv file) are: tweet ID, ISO country
code, tweet date, tweet text, definitely English, ambiguous, definitely not
English, code-switched, ambiguous due to named entities, and automatically
generated tweets. 

All annotations are binary; the definitely English, ambiguous, and definitely
not English annotations are mutually exclusive. Experiments reported in our
paper were run by excluding tweets labeled as ambiguous or automatically
generated.  

Our paper also describes a language classifier, which is available on our
website: http://slanglab.cs.umass.edu/TwitterLangID/

Our data annotations are licensed under the Creative Commons Attribution 4.0
International License: https://creativecommons.org/licenses/by/4.0/
