# pdxFilter

Filter out mouse reads from patient derived xenografts

This project is broken into four main parts:

## Part 1 - Preprocessing
If you have a known positive and negative data source, you can run

        'python3 sample_reads.py -d pos.fastq neg.fastq'

which will return a pos.txt and neg.txt file containing sampled reads
If you have a single unknown (mixed/contaminated) data sources, you can run

        'python3 sample_reads.py -s unknown.fastq'

which will return an unknown.txt file conftaining sampled reads
Run the command 

        'python3 sample_reads.py --help'

 to see aditional optional preprocessing parameters
      
## Part 2 - Training
Training is optional, as you can restore pre-trained models
If you would like to retrain a model, this requires data from a known positive and negative source that is pre-processed using the command listed in part 1.
The default input files for training are 'pos.txt' and 'neg.txt', but these can be altered in lines 15 and 16:

    'tf.flags.DEFINE_string("positive_data_file", "pos.txt", "Data source for positive data (human reads)")'
    'tf.flags.DEFINE_string("negative_data_file", "neg.txt", "Data source for negative data (mouse reads)")'

by changing the 'pos.txt' and 'neg.txt' fields to your corresponding preprocessed files

## Part 3 - Evaluation/Prediction
Evaluation can be used to evaluate a model, or to generate predictions for an unknown dataset.
To evaluate a model on labeled data, simply run 

      'python3 eval.py' 

This will restore the most recent trained model, and will then procede to evaluate the models performance on the positive and negative data fields specified in like 13 and 14. The default data fields are once again pos.txt and neg.txt
To generate predictions on unknown data, change the value for the eval_unkown flag on line 24 from False, to True.  This will write out the models classification predictions for the reads contained in the file unknown.txt.  This file name can be changed on line 18 of eval.py.  The predictions will be written out to 'predictions.csv'

## Part 4 - Filtering
We can filter our predictions based on the confidence of each prediction.  Running the command:

     'python3 filter.py'

will take a `~/pdxFilter/predictions.csv` file, and return a file filtered_reads.fastq.  This file will remove all negative reads from the unknown data s\
et that are above a certain confidence threshold.

## Dependancies
        Tensorflow
        Biopython (Bio.SeqIO, Bio.ParseIO)
        Numpy
        Pandas
        ParseFastQ

## Citations/Aknowledgement

Used bag of word model for CNN input as described in this paper:
        Nguyen, N.G., Tran, V.A., Ngo, D.L., Phan, D., Lumbanraja, F.R., Faisal, M.R., Abapihi, B., Kubo, M. and Satou, K. (2016) DNA Sequence Classifica\
tion by Convolutional Neural Network. J. Biomedical Science and Engineering, 9, 280-286. http://dx.doi.org/10.4236/jbise.2016.95021

Adapted much of the CNN code from this tensorflow blogpost tutorial
        Britz, Denny. "Implementing a CNN for Text Classification in TensorFlow." Web log post. WILDML. N.p., 11 Dec. 2015. Web. 1 Aug. 2017
