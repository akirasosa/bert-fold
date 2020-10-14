# BERT meets the Protein 3D structure

As we know, BERT is commonly used in NLP. It has made remarkable results ever. Recently, some researches are reported, which apply BERT to solve some tasks for Protein.

In this article, I introduce how the protein is related with BERT. Furthermore, I will show my own experiment to solve the protein structure, which is one of the most difficult, but important task in biochemistry.

## Protein is a sequence

Protein is a sequence of amino acid. There are 20 standard amino acids such as Alanine, Arginine and etc. They are chained from N-term to C-term by peptide bond.

![](images/736px-Protein_primary_structure.svg.png)

The actual structure of the protein is not like a chain, but it can have various 3D structures like below.

![](images/lysozyme.jpeg)
![](images/2dt5_redox_sensor.jpeg)
![](images/1k4r_assembly-1.jpeg)

The left most is the protein called as Lysozyme. You can see that the single chain is folded and consists a 3D structure. Lysozyme is an enzyme. The pocket at the middle plays an important role to grab the specific substrate.

The middle is the transcription regulator protein (and its structure was solved by me). You can see that 2 identical chains are entangled with one another. It suits to bind with DNA. 

The right most is the Dengue Virus. It's also made from proteins.

As we have seen above, the 3D structure is important for the protein to play its functionality. Such structure is implied by the sequence of the amino acids. That's the main idea.

It usually costs a lot to determine 3D structure of protein. So, it's wonderful, if we can know the functionality and structure of the protein from only the sequence information.

## Machine learning tasks in protein 

Protein is a sequence. There are a lot of proteins existing in the world. It's not so difficult to know the sequence itself. It reminds us the BERT.

Actually, Ahmed et al.released ProtTrans recently. ProtTrans is the collection of various transformer models which are pre-trained with 217 million protein sequences. As same as BERT in NLP, they are trained by MLM.

They also publishes the results of some downstream tasks such as Secondary Structure Prediction and Membrane-bound vs Water-soluble.

Besides the BERT, the machine learning has been already used for some protein tasks. 
AlphaFold would be the most famous one. DeepMind has developed AlphaFold to solve the task of predicting Protein 3D structure at 2018. It wins the CASP13 which is a competition held for each 2 years.

Unfortunately, the source code of AlphaFold is not published. Instead, community-built, open source implementation, is published [here](https://github.com/dellacortelab/prospr). We can see some results of distance map predictions.

![](./images/T0954.jpeg)

## BertFold - My own experiment

We have a pre-trained BERT model for protein. How well does it work for predicting 3D structure? It's natural thought. So, I have tried it.