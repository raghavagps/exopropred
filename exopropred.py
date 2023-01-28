#########################################################################################
# ExoProPred is developed for predicting the exosomal proteins using hybrid model which #
# combines ML & Motif search approach. It is developed by Prof G. P. S. Raghava's group.#
# Please cite: https://webs.iiitd.edu.in/raghava/exopropred/                           #
########################################################################################
import argparse
import warnings
import subprocess
import pkg_resources
import os
import sys
import numpy as np
import pandas as pd
import math
import itertools
from collections import Counter
import pickle
import re
import glob
import time
import uuid
from time import sleep
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesClassifier
import urllib.request
import shutil
import zipfile
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Please provide following arguments') 

## Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence(s) in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output",type=str, help="Output: File for saving results by default outfile.csv")
parser.add_argument("-m", "--model",type=int, choices = [1,2], help="Model Type: 1: Composition based model, 2: Hybrid Model, by default 1")
parser.add_argument("-t","--threshold", type=float, help="Threshold: Value between 0 to 1 by default 0.51")
parser.add_argument("-d","--display", type=int, choices = [1,2], help="Display: 1:Exosomal Proteins only, 2: All Proteins, by default 1")
args = parser.parse_args()

# Function to check the seqeunce
def readseq(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', ''.join(array[1:]).upper())
        seqid.append('>'+name)
        seq.append(sequence)
    if len(seqid) == 0:
        f=open(file,"r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n',''))
        for i in range (1,len(seq)+1):
            seqid.append(">Seq_"+str(i))
    df1 = pd.DataFrame(seqid)
    df2 = pd.DataFrame(seq)
    return df1,df2
# Function to check the length of the input sequences
def lenchk(file):
    dflc = file
    dflc.columns = ['Seq']
    for i in range(len(dflc)):
        if len(dflc['Seq'][i])<50:
            print("###########################################################################################################################################################")
            print("Error: Please provide sequence(s) with length more than 50. Input sequence at position "+i+" has length "+len(dflc['Seq'][i])+". Please check the sequences.")
            print("###########################################################################################################################################################")
            sys.exit()
        else:
            continue    
# Function to split files
def file_split(file):
    df_2,df_3 = readseq(file)
    df1 = pd.concat([df_2,df_3],axis=1)
    df1.columns = ['ID','Seq']
    if os.path.isdir('fasta') == False:
        os.mkdir(os.getcwd()+'/fasta')
    else:
        pass
    path = os.getcwd()+'/fasta'
    for i in range(len(df1)):
        df1.loc[i].to_csv(path+'/'+df1['ID'][i].replace('>','')+'.fasta', index=None,header=False,sep="\n")

# Function to generate PSSM files
def pssm_gen(dir_path,pspath,sppath):
    listdir = glob.glob(dir_path+'/fasta/*.fasta')
    if os.path.isdir('pssm_raw') == False:
        os.mkdir(dir_path+'/pssm_raw')
    else:
        pass
    if os.path.isdir('pssm_raw1') == False:
        os.mkdir(dir_path+'/pssm_raw1')
    else:
        pass
    for i in listdir:
        filename = i.split('/')[-1].split('.')[0]
        cmd = pspath+" -out "+dir_path+"/pssm_raw1/"+filename+".homologs -outfmt 7 -query "+dir_path+"/fasta/"+filename+".fasta -db "+sppath+" -evalue 0.0001 -word_size 3 -max_target_seqs 6000 -num_threads 4 -gapopen 11 -gapextend 1 -matrix BLOSUM62 -comp_based_stats T -num_iterations 3 -out_pssm "+dir_path+"/pssm_raw1/"+filename+".cptpssm -out_ascii_pssm "+dir_path+"/pssm_raw/"+filename+".pssm"
        os.system(cmd)
    filelist = [f for f in os.listdir(dir_path+"/pssm_raw1/") if f.endswith(".homologs") or f.endswith(".cptpssm")]
    for f in filelist:
        os.remove(dir_path+"/pssm_raw1/"+f)
    os.rmdir(dir_path+"/pssm_raw1")
# Function to generate the features out of seqeunces
PCP= pd.read_csv(os.path.dirname(__file__)+'Data/PhysicoChemical.csv', header=None)
AAIndex = pd.read_csv(os.path.dirname(__file__)+'Data/aaindex.csv',index_col='INDEX');
AAIndexNames = pd.read_csv(os.path.dirname(__file__)+'Data/AAIndexNames.csv',header=None);
std = list('ACDEFGHIKLMNPQRSTVWY')
def aac_comp(file):
    df1 = pd.read_csv(file, header=None, names=['Seq'])
    dd = []
    for j in df1['Seq']:
        cc = []
        for i in std:
            count = 0
            for k in j:
                temp1 = k
                if temp1 == i:
                    count += 1
                composition = (count/len(j))*100
            cc.append(composition)
        dd.append(cc)
    df2 = pd.DataFrame(dd)
    head = []
    for mm in std:
        head.append('AAC_'+mm)
    df2.columns = head
    return df2
def atc(file):
    atom=pd.read_csv(os.path.dirname(__file__)+"Data/atom.csv",header=None)
    at=pd.DataFrame()
    i = 0
    C_atom = []
    H_atom = []
    N_atom = []
    O_atom = []
    S_atom = []

    while i < len(atom):
        C_atom.append(atom.iloc[i,1].count("C"))
        H_atom.append(atom.iloc[i,1].count("H"))
        N_atom.append(atom.iloc[i,1].count("N"))
        O_atom.append(atom.iloc[i,1].count("O"))
        S_atom.append(atom.iloc[i,1].count("S"))
        i += 1
    atom["C_atom"]=C_atom
    atom["O_atom"]=O_atom
    atom["H_atom"]=H_atom
    atom["N_atom"]=N_atom
    atom["S_atom"]=S_atom
    test1 = pd.read_csv(file,header=None)
    dd = []
    for i in range(0, len(test1)):
        dd.append(test1[0][i].upper())
    test = pd.DataFrame(dd)
    count_C = 0
    count_H = 0
    count_N = 0
    count_O = 0
    count_S = 0
    count = 0
    i1 = 0
    j = 0
    k = 0
    C_ct = []
    H_ct = []
    N_ct = []
    O_ct = []
    S_ct = []
    while i1 < len(test) :
        while j < len(test[0][i1]) :
            while k < len(atom) :
                if test.iloc[i1,0][j]==atom.iloc[k,0].replace(" ","") :
                    count_C = count_C + atom.iloc[k,2]
                    count_H = count_H + atom.iloc[k,3]
                    count_N = count_N + atom.iloc[k,4]
                    count_O = count_O + atom.iloc[k,5]
                    count_S = count_S + atom.iloc[k,6]
                #count = count_C + count_H + count_S + count_N + count_O
                k += 1
            k = 0
            j += 1
        C_ct.append(count_C)
        H_ct.append(count_H)
        N_ct.append(count_N)
        O_ct.append(count_O)
        S_ct.append(count_S)
        count_C = 0
        count_H = 0
        count_N = 0
        count_O = 0
        count_S = 0
        j = 0
        i1 += 1
    test["C_count"]=C_ct
    test["H_count"]=H_ct
    test["N_count"]=N_ct
    test["O_count"]=O_ct
    test["S_count"]=S_ct

    ct_total = []
    m = 0
    while m < len(test) :
        ct_total.append(test.iloc[m,1] + test.iloc[m,2] + test.iloc[m,3] + test.iloc[m,4] + test.iloc[m,5])
        m += 1
    test["count"]=ct_total
##########final output#####
    final = pd.DataFrame()
    n = 0
    p = 0
    C_p = []
    H_p = []
    N_p = []
    O_p = []
    S_p = []
    while n < len(test):
        C_p.append((test.iloc[n,1]/test.iloc[n,6])*100)
        H_p.append((test.iloc[n,2]/test.iloc[n,6])*100)
        N_p.append((test.iloc[n,3]/test.iloc[n,6])*100)
        O_p.append((test.iloc[n,4]/test.iloc[n,6])*100)
        S_p.append((test.iloc[n,5]/test.iloc[n,6])*100)
        n += 1
    final["ATC_C"] = C_p
    final["ATC_H"] = H_p
    final["ATC_N"] = N_p
    final["ATC_O"] = O_p
    final["ATC_S"] = S_p
    return final.round(2)
def val(AA_1, AA_2, aa, mat):
    return sum([(mat[i][aa[AA_1]] - mat[i][aa[AA_2]]) ** 2 for i in range(len(mat))]) / len(mat)
def paac_1(file,lambdaval=1,w=0.05):
    data1 = pd.read_csv(os.path.dirname(__file__)+"Data/data", sep = "\t")
    filename, file_extension = os.path.splitext(file)
    df = pd.read_csv(file, header = None)
    df1 = pd.DataFrame(df[0].str.upper())
    dd = []
    cc = []
    pseudo = []
    aa = {}
    for i in range(len(std)):
        aa[std[i]] = i
    for i in range(0,3):
        mean = sum(data1.iloc[i][1:])/20
        rr = math.sqrt(sum([(p-mean)**2 for p in data1.iloc[i][1:]])/20)
        dd.append([(p-mean)/rr for p in data1.iloc[i][1:]])
        zz = pd.DataFrame(dd)
    head = []
    for n in range(1, lambdaval + 1):
        head.append('_lam' + str(n))
    head = ['PAAC'+str(lambdaval)+sam for sam in head]
    pp = pd.DataFrame()
    ee = []
    for k in range(0,len(df1)):
        cc = []
        pseudo1 = []
        for n in range(1,lambdaval+1):
            cc.append(sum([val(df1[0][k][p], df1[0][k][p + n], aa, dd) for p in range(len(df1[0][k]) - n)]) / (len(df1[0][k]) - n))
            qq = pd.DataFrame(cc)
        pseudo = pseudo1 + [(w * p) / (1 + w * sum(cc)) for p in cc]
        ee.append(pseudo)
        ii = round(pd.DataFrame(ee, columns = head),4)
    return ii
def paac(file,lambdaval=1,w=0.05):
    data2 = paac_1(file,lambdaval=1,w=0.05)
    data1 = aac_comp(file)
    header = ['PAAC'+str(lambdaval)+'_A','PAAC'+str(lambdaval)+'_C','PAAC'+str(lambdaval)+'_D','PAAC'+str(lambdaval)+'_E','PAAC'+str(lambdaval)+'_F','PAAC'+str(lambdaval)+'_G','PAAC'+str(lambdaval)+'_H','PAAC'+str(lambdaval)+'_I','PAAC'+str(lambdaval)+'_K','PAAC'+str(lambdaval)+'_L','PAAC'+str(lambdaval)+'_M','PAAC'+str(lambdaval)+'_N','PAAC'+str(lambdaval)+'_P','PAAC'+str(lambdaval)+'_Q','PAAC'+str(lambdaval)+'_R','PAAC'+str(lambdaval)+'_S','PAAC'+str(lambdaval)+'_T','PAAC'+str(lambdaval)+'_V','PAAC'+str(lambdaval)+'_W','PAAC'+str(lambdaval)+'_Y']
    data1.columns = header
    data3 = pd.concat([data1,data2], axis = 1).reset_index(drop=True)
    return data3.round(2)
def qso(file,gap=1,w=0.1):
    ff = []
    df = pd.read_csv(file, header = None)
    df2 = pd.DataFrame(df[0].str.upper())
    for i in range(0,len(df2)):
        ff.append(len(df2[0][i]))
    if min(ff) < gap:
        print("Error: All sequences' length should be higher than :", gap)
    else:
        mat1 = pd.read_csv(os.path.dirname(__file__)+"Data/Schneider-Wrede.csv", index_col = 'Name')
        mat2 = pd.read_csv(os.path.dirname(__file__)+"Data/Grantham.csv", index_col = 'Name')
        s1 = []
        s2 = []
        for i in range(0,len(df2)):
            for n in range(1, gap+1):
                sum1 = 0
                sum2 = 0
                for j in range(0,(len(df2[0][i])-n)):
                    sum1 = sum1 + (mat1[df2[0][i][j]][df2[0][i][j+n]])**2
                    sum2 = sum2 + (mat2[df2[0][i][j]][df2[0][i][j+n]])**2
                s1.append(sum1)
                s2.append(sum2)
        zz = pd.DataFrame(np.array(s1).reshape(len(df2),gap))
        zz["sum"] = zz.sum(axis=1)
        zz2 = pd.DataFrame(np.array(s2).reshape(len(df2),gap))
        zz2["sum"] = zz2.sum(axis=1)
        c1 = []
        c2 = []
        c3 = []
        c4 = []
        h1 = []
        h2 = []
        h3 = []
        h4 = []
        for aa in std:
            h1.append('QSO'+str(gap)+'_SC_' + aa)
        for aa in std:
            h2.append('QSO'+str(gap)+'_G_' + aa)
        for n in range(1, gap+1):
            h3.append('SC' + str(n))
        h3 = ['QSO'+str(gap)+'_'+sam for sam in h3]
        for n in range(1, gap+1):
            h4.append('G' + str(n))
        h4 = ['QSO'+str(gap)+'_'+sam for sam in h4]
        for i in range(0,len(df2)):
            AA = {}
            for j in std:
                AA[j] = df2[0][i].count(j)
                c1.append(AA[j] / (1 + w * zz['sum'][i]))
                c2.append(AA[j] / (1 + w * zz2['sum'][i]))
            for k in range(0,gap):
                c3.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))
                c4.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))
        pp1 = np.array(c1).reshape(len(df2),len(std))
        pp2 = np.array(c2).reshape(len(df2),len(std))
        pp3 = np.array(c3).reshape(len(df2),gap)
        pp4 = np.array(c4).reshape(len(df2),gap)
        zz5 = round(pd.concat([pd.DataFrame(pp1, columns = h1),pd.DataFrame(pp2,columns = h2),pd.DataFrame(pp3, columns = h3),pd.DataFrame(pp4, columns = h4)], axis = 1),4)
        return zz5.round(2)
def SER(filename):
    data=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
    data2=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
    Val=np.zeros(len(data))
    GH=[]
    for i in range(len(data)):
        my_list={'A':0,'C':0,'D':0,'E':0,'F':0,'G':0,'H':0,'I':0,'K':0,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':0,'S':0,'T':0,'V':0,'W':0,'Y':0}
        data1=''
        data1=str(data[i])
        data1=data1.upper()
        allowed = set(('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'))
        is_data_invalid = set(data1).issubset(allowed)
        if is_data_invalid==False:
            print("Error: Please check for invalid inputs in the sequence.","\nError in: ","Sequence number=",i+1,",","Sequence = ",data[i],",","\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
            return
        seq=data[i]
        seq=seq.upper()
        num, length = Counter(seq), len(seq)
        num=dict(sorted(num.items()))
        C=list(num.keys())
        F=list(num.values())
        for key, value in my_list.items():
             for j in range(len(C)):
                if key == C[j]:
                    my_list[key] = -round(((F[j]/length)* math.log(F[j]/length, 2)),3)
        GH.append(list(my_list.values()))
        df = pd.DataFrame(GH)
        df.columns = ['SER_A','SER_C','SER_D','SER_E','SER_F','SER_G','SER_H','SER_I','SER_K','SER_L','SER_M','SER_N','SER_P','SER_Q','SER_R','SER_S','SER_T','SER_V','SER_W','SER_Y']
    return df
def encode(peptide):
    l=len(peptide);
    encoded=np.zeros(l);
    for i in range(l):
        if(peptide[i]=='A'):
            encoded[i] = 0;
        elif(peptide[i]=='C'):
            encoded[i] = 1;
        elif(peptide[i]=='D'):
            encoded[i] = 2;
        elif(peptide[i]=='E'):
            encoded[i] = 3;
        elif(peptide[i]=='F'):
            encoded[i] = 4;
        elif(peptide[i]=='G'):
            encoded[i] = 5;
        elif(peptide[i]=='H'):
            encoded[i] = 6;
        elif(peptide[i]=='I'):
            encoded[i] = 7;
        elif(peptide[i]=='K'):
            encoded[i] = 8;
        elif(peptide[i]=='L'):
            encoded[i] = 9;
        elif(peptide[i]=='M'):
            encoded[i] = 10;
        elif(peptide[i]=='N'):
            encoded[i] = 11;
        elif(peptide[i]=='P'):
            encoded[i] = 12;
        elif(peptide[i]=='Q'):
            encoded[i] = 13;
        elif(peptide[i]=='R'):
            encoded[i] = 14;
        elif(peptide[i]=='S'):
            encoded[i] = 15;
        elif(peptide[i]=='T'):
            encoded[i] = 16;
        elif(peptide[i]=='V'):
            encoded[i] = 17;
        elif(peptide[i]=='W'):
            encoded[i] = 18;
        elif(peptide[i]=='Y'):
            encoded[i] = 19;
        else:
            print('Wrong residue!');
    return encoded;
def lookup(peptide,featureNum):
    l=len(peptide);
    peptide = list(peptide);
    out=np.zeros(l);
    peptide_num = encode(peptide);
    for i in range(l):
        out[i] = PCP[peptide_num[i]][featureNum];
    return sum(out);
def pcp(file):
    SEP_headers = ['SEP_PC','SEP_NC','SEP_NE','SEP_PO','SEP_NP','SEP_AL','SEP_CY','SEP_AR','SEP_AC','SEP_BS','SEP_NE_pH','SEP_HB','SEP_HL','SEP_NT','SEP_HX','SEP_SC','SEP_SS_HE','SEP_SS_ST','SEP_SS_CO','SEP_SA_BU','SEP_SA_EX','SEP_SA_IN','SEP_TN','SEP_SM','SEP_LR']
    if(type(file) == str):
        seq = pd.read_csv(file,header=None, sep=',');
        seq=seq.T
        seq[0].values.tolist()
        seq=seq[0];
    else:
        seq  = file;
    l = len(seq);
    rows = PCP.shape[0]; # Number of features in our reference table
    col = 20 ; # Denotes the 20 amino acids
    seq=[seq[i].upper() for i in range(l)]
    sequenceFeature = [];
    sequenceFeature.append(SEP_headers); #To put property name in output csv

    for i in range(l): # Loop to iterate over each sequence
        nfeatures = rows;
        sequenceFeatureTemp = [];
        for j in range(nfeatures): #Loop to iterate over each feature
            featureVal = lookup(seq[i],j)
            if(len(seq[i])!=0):
                sequenceFeatureTemp.append(featureVal/len(seq[i]))
            else:
                sequenceFeatureTemp.append('NaN')
        sequenceFeature.append(sequenceFeatureTemp);
    out = pd.DataFrame(sequenceFeature);
    return sequenceFeature;
def phyChem(file,mode='all',m=0,n=0):
    if(type(file) == str):
        seq1 = pd.read_csv(file,header=None, sep=',');
        seq1 = pd.DataFrame(seq1[0].str.upper())
        seq=[]
        [seq.append(seq1.iloc[i][0]) for i in range(len(seq1))]
    else:
        seq  = file;
    l = len(seq);
    newseq = [""]*l; # To store the n-terminal sequence
    for i in range(0,l):
        l = len(seq[i]);
        if(mode=='NT'):
            n=m;
            if(n!=0):
                newseq[i] = seq[i][0:n];
            elif(n>l):
                print('Warning! Sequence',i,"'s size is less than n. The output table would have NaN for this sequence");
            else:
                print('Value of n is mandatory, it cannot be 0')
                break;
        elif(mode=='CT'):
            n=m;
            if(n!=0):
                newseq[i] = seq[i][(len(seq[i])-n):]
            elif(n>l):
                print('WARNING: Sequence',i+1,"'s size is less than the value of n given. The output table would have NaN for this sequence");
            else:
                print('Value of n is mandatory, it cannot be 0')
                break;
        elif(mode=='all'):
            newseq = seq;
        elif(mode=='rest'):
            if(m==0):
                print('Kindly provide start index for rest, it cannot be 0');
                break;
            else:
                if(n<=len(seq[i])):
                    newseq[i] = seq[i][m-1:n+1]
                elif(n>len(seq[i])):
                    newseq[i] = seq[i][m-1:len(seq[i])]
                    print('WARNING: Since input value of n for sequence',i+1,'is greater than length of the protein, entire sequence starting from m has been considered')
        else:
            print("Wrong Mode. Enter 'NT', 'CT','all' or 'rest'");
    output = pcp(newseq);
    return output
def shannons(filename):
    SEP_headers = ['SEP_PC','SEP_NC','SEP_NE','SEP_PO','SEP_NP','SEP_AL','SEP_CY','SEP_AR','SEP_AC','SEP_BS','SEP_NE_pH','SEP_HB','SEP_HL','SEP_NT','SEP_HX','SEP_SC','SEP_SS_HE','SEP_SS_ST','SEP_SS_CO','SEP_SA_BU','SEP_SA_EX','SEP_SA_IN','SEP_TN','SEP_SM','SEP_LR']
    if(type(filename) == str):
        seq1 = pd.read_csv(filename,header=None, sep=',');
        seq1 = pd.DataFrame(seq1[0].str.upper())
    else:
        seq1  = filename;
    seq=[]
    [seq.append(seq1.iloc[i][0]) for i in range(len(seq1))]
    comp = phyChem(seq);
    new = [comp[i][0:25] for i in range(len(comp))]
    entropy  = [];
    entropy.append(SEP_headers[0:25])
    for i in range(1,len(new)):
        seqEntropy = [];
        for j in range(len(new[i])):
            p = new[i][j];
            if((1-p) == 0. or p==0.):
                temp = 0;#to store entropy of each sequence
            else:
                temp = -(p*math.log2(p)+(1-p)*math.log2(1-p));
            seqEntropy.append(round(temp,3));
        entropy.append(seqEntropy);
    out = pd.DataFrame(entropy);
    out.columns = out.loc[0]
    out = out.loc[1:]
    out.reset_index(drop=True, inplace=True)
    return out;
def searchAAIndex(AAIndex):
    found = -1;
    for i in range(len(AAIndexNames)):
        if(str(AAIndex) == AAIndexNames.iloc[i][0]):
            found = i;
    return found;
def phychem_AAI(file):
    AAIn = ['AAI_BURA740102','AAI_CRAJ730103','AAI_GEOR030104','AAI_PALJ810101','AAI_PLIV810101','AAI_VINM940101','AAI_WERD780103']
    seq = pd.read_csv(file,header=None);
    seq = seq[0]
    AAI = AAIn;
    l2 = len(AAI)
    header  = AAI[0:l2];
    final=[];
    final.append(AAI);
    l1 = len(seq);
    seq=[seq[i].upper() for i in range(l1)];
    for i in range(l1):
        coded = encode(seq[i]);
        temp=[];
        for j in range(l2):
            pos = searchAAIndex(AAI[j]);
            sum=0;
            for k in range(len(coded)):
                val = AAIndex.iloc[pos,int(coded[k])]
                sum=sum+val;
            avg = round(sum/len(seq[i]),3);
            temp.append(avg);
        final.append(temp);
    out = pd.DataFrame(final);
    out.columns = out.loc[0]
    out = out.loc[1:]
    out.reset_index(drop=True, inplace=True)
    return out;
def feature_gen(file):
    df1 = phychem_AAI(file)
    df2 = atc(file)
    df3 = paac(file)
    df4 = qso(file)
    df5 = SER(file)
    df6 = shannons(file)
    df7 = pd.concat([df1,df2,df3,df4,df5,df6], axis=1)
    return df7
# For Top Features
def top_feat(file1,file2,file3,file4):
    df1 = file1
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)
    df5 = pd.concat([df1,df2,df3,df4],axis=1)
    feat = ['pssm_composition4','pssm_composition7','pssm_composition10','pssm_composition13','pssm_composition19','pssm_composition23','pssm_composition28','pssm_composition39','pssm_composition49','pssm_composition61','pssm_composition66','pssm_composition69','pssm_composition71','pssm_composition87','pssm_composition89','pssm_composition99','pssm_composition101','pssm_composition124','pssm_composition128','pssm_composition143','pssm_composition153','pssm_composition168','pssm_composition179','pssm_composition213','pssm_composition217','pssm_composition218','pssm_composition228','pssm_composition232','pssm_composition237','pssm_composition259','pssm_composition268','pssm_composition295','pssm_composition298','pssm_composition301','pssm_composition303','pssm_composition306','pssm_composition315','pssm_composition319','pssm_composition325','pssm_composition329','pssm_composition330','pssm_composition337','pssm_composition339','pssm_composition364','pssm_composition394','pssm_composition398','pssm_composition399','aac_pssm11','aac_pssm15','dpc_pssm239','AAI_BURA740102','AAI_CRAJ730103','AAI_GEOR030104','AAI_PALJ810101','AAI_PLIV810101','AAI_VINM940101','AAI_WERD780103','ATC_N','ATC_S','PAAC1_W','QSO1_SC_I','QSO1_SC_W','QSO1_G1','SER_L','SER_N','SER_W','SEP_PC','SEP_AR','SEP_BS','SEP_SA_EX']
    df6 = df5[feat]
    return df6
# Function to process the MERCI output
def BLAST_processor(blast_result,name1,ml_results,thresh):
    if os.stat(blast_result).st_size != 0:
        df1 = pd.read_csv(blast_result, sep="\t", names=['name','hit','identity','r1','r2','r3','r4','r5','r6','r7','r8','r9'])
        df__2 = name1
        df2 = pd.DataFrame()
        df2 = df2.append(df__2.values.tolist())
        df3 = ml_results
        cc = []
        for i in df2[0]:
            kk = i.replace('>','')
            if len(df1.loc[df1.name==kk])>0:
                df4 = df1[['name','hit']].loc[df1['name']==kk].reset_index(drop=True)
                if df4['hit'][0].split('_')[0]=='P':
                    cc.append(0.5)
                if df4['hit'][0].split('_')[0]=='N':
                    cc.append(-0.5)
            else:
                cc.append(0)
        df6 = pd.DataFrame()
        df6['Seq_ID'] = [i.replace('>','') for i in df2[0]]
        df6['ML_Score'] = df3['ML_score']
        df6['BLAST_Score'] = cc
        df6['Total_Score'] = df6['ML_Score']+df6['BLAST_Score']
        df6['Prediction'] = ['Binder' if df6['Total_Score'][i]>thresh else 'Non-binder' for i in range(0,len(df6))]
    else:
        df__2 = name1
        df3 = ml_results
        df2 = pd.DataFrame()
        df2 = df2.append(df__2.values.tolist())
        ss = []
        vv = []
        for j in df2[0]:
            ss.append(j.replace('>',''))
            vv.append(0)
        df6 = pd.DataFrame()
        df6['Seq_ID'] = ss
        df6['ML_Score'] = df3['ML_score']
        df6['BLAST_Score'] = vv
        df6['Total_Score'] = df6['ML_Score']+df6['BLAST_Score']
        df6['Prediction'] = ['Binder' if df6['Total_Score'][i]>thresh else 'Non-binder' for i in range(0,len(df6))]
    return df6
# Function to read and implement the model
def model_run(file1,file2):
    a = []
    data_test = file1
    clf = pickle.load(open(file2,'rb'))
    y_p_score1=clf.predict_proba(data_test)
    y_p_s1=y_p_score1.tolist()
    a.extend(y_p_s1)
    df = pd.DataFrame(a)
    df1 = df.iloc[:,-1].round(2)
    df2 = pd.DataFrame(df1)
    df2.columns = ['ML_score']
    return df2
# Functions to locate motifs
def motif_locate(file):
    cc = []
    with open(file, 'r') as f:
        fob = f.readlines()
        for line in fob:
            if 'motifs match' in line:
                cc.append('>'+line.split(' ')[0])
    return cc
# Functions to assign weights to the queries
def motif_wt(file,pm,nm):
    cc = []
    dd = []
    for i in file.ID:
        if i in pm:
            cc.append(0.5)
        else:
            cc.append(0)
    for j in file.ID:
        if j in nm:
            dd.append(-0.5)
        else:
            dd.append(0)
    return cc,dd
# Function to processing merci outputs
def merci_processor(file1,file2,file3,file4,file5):
    df_tr_1, df_tr_2 = readseq(file1)
    df_train = pd.concat([df_tr_1,df_tr_2],axis=1) 
    df_train.columns = ['ID','Seq']
    df_ML_train = file2
    df_111 = pd.concat([df_train,df_ML_train],axis=1)
    tr_p = motif_locate(file3)
    tr_n = motif_locate(file4)
    trm, trn = motif_wt(df_111,tr_p,tr_n)
    df_111['Pos'] = trm
    df_111['Neg'] = trn
    df_111['Overall_Score'] = df_111['ML_score']+df_111['Pos']+df_111['Neg']
    df_111 = df_111.round(2)
    xx = []
    for i in range(len(df_111)):
        if df_111['Overall_Score'][i] >= file5:
            xx.append('Exosomal')
        else:
            xx.append('Non-Exosomal')
    df_111['Prediction'] = xx
    df_22 = df_111[['ML_score','Pos','Neg','Overall_Score','Prediction']]
    return df_22
('############################################################################################')
print('# This program ExoProPred is developed for predicting the exosomal proteins among the #')
print('# submitted protein sequence(s), developed by Prof G. P. S. Raghava group.            #')
print('# Please cite: ExoProPred; available at https://webs.iiitd.edu.in/raghava/exopropred/ #')
print('#######################################################################################')

# Parameter initialization or assigning variable for command level arguments

Sequence= args.input        # Input variable 
 
# Output file 
 
if args.output == None:
    result_filename= "outfile.csv" 
else:
    result_filename = args.output
         
# Threshold 
if args.threshold == None:
        Threshold = 0.51
else:
        Threshold= float(args.threshold)
# Job Type 
if args.model == None:
        Model = int(1)
else:
        Model = int(args.model)
# Display
if args.display == None:
        dplay = int(1)
else:
        dplay = int(args.display)


#####################################BLAST Path############################################
if os.path.exists('envfile'):
    with open('envfile', 'r') as file:
        data = file.readlines()
    output = []
    for line in data:
        if not "#" in line:
            output.append(line)
    if len(output)==4:
        paths = []
        for i in range (0,len(output)):
            paths.append(output[i].split(':')[1].replace('\n',''))
        psiblast = paths[0]
        merci = paths[1]
        motifs = paths[2]
        swiss = paths[3]
    else:
        print("##############################################################################################################")
        print("Error: Please provide paths for PSI-BLAST, Swiss-Prot Database, and MERCI, and required files", file=sys.stderr)
        print("##############################################################################################################")
        sys.exit()
else:
    print("######################################################################################################")
    print("Error: Please provide the '{}', which comprises paths for PSI-BLAST".format('envfile'), file=sys.stderr)
    print("######################################################################################################")
    sys.exit()
###########################################################################################

print("\n");
print('##############################################################################')
print('Summary of Parameters:')
print('Input File: ',Sequence,'; Threshold: ', Threshold,'; Model Type: ',Model)
print('Output File: ',result_filename,'; Display: ',dplay)
print('# ############################################################################')
#========================================Extracting Model====================================
if os.path.isdir('model') == False:
    with zipfile.ZipFile('./model.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
else:
    pass
#=======================================SwissProt Database Download===========================
if os.path.isdir('swissprot') == False:
    urllib.request.urlretrieve("https://webs.iiitd.edu.in/raghava/exopropred/swissprot.zip","swissprot.zip")
    with zipfile.ZipFile('./swissprot.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove('./swissprot.zip')
else:
    pass
#======================= Prediction Module start from here =====================
if Model == 1:
    print('\n======= Thanks for using Predict module of ExoProPred using composition based model. Your results will be stored in file :',result_filename,' =====\n')
    df_2,df1 = readseq(Sequence)
    lenchk(df1)
    filename = str(uuid.uuid4())
    filename_1 = str(uuid.uuid4())
    df1.to_csv(filename_1,index=None,header=False)
    feat = feature_gen(filename_1)
    df11 = pd.concat([df_2,df1],axis=1)
    df11.to_csv(filename,index=None,header=False,sep="\n")
    file_split(Sequence)
    pssm_gen(os.getcwd(),psiblast,swiss)
    os.system("python3 "+os.path.dirname(__file__)+"src/possum.py -i "+filename+" -o "+filename+"_example_aac_pssm_no_header -t aac_pssm -p "+os.getcwd()+"/pssm_raw -a 0 -b 0");
    os.system("python3 "+os.path.dirname(__file__)+"src/headerHandler.py -i "+filename+"_example_aac_pssm_no_header -p aac_pssm -n 20 -o "+filename+"_aac_pssm.csv");
    os.system("python3 "+os.path.dirname(__file__)+"src/possum.py -i "+filename+" -o "+filename+"_example_dpc_pssm_no_header -t dpc_pssm -p "+os.getcwd()+"/pssm_raw -a 5 -b 0");
    os.system("python3 "+os.path.dirname(__file__)+"src/headerHandler.py -i "+filename+"_example_dpc_pssm_no_header -p dpc_pssm -n 400 -o "+filename+"_dpc_pssm.csv");
    os.system("python3 "+os.path.dirname(__file__)+"src/possum.py -i "+filename+" -o "+filename+"_example_pssm_comp_no_header -t pssm_composition -p "+os.getcwd()+"/pssm_raw -a 5 -b 0");
    os.system("python3 "+os.path.dirname(__file__)+"src/headerHandler.py -i "+filename+"_example_pssm_comp_no_header -p pssm_composition -n 400 -o "+filename+"_pssm_comp.csv");
    X = top_feat(feat,filename+"_aac_pssm.csv",filename+"_dpc_pssm.csv",filename+"_pssm_comp.csv")
    mlres = model_run(X,'model/RF_model.pkl')
    mlres = mlres.round(3)
    df44 = pd.concat([df_2,mlres],axis=1)
    df44['Prediction'] = ['Exosomal' if df44['ML_score'][i]>=Threshold else 'Non-exosomal' for i in range(len(df44))]
    df44.columns = ['Seq_ID','Score','Prediction']
    if dplay == 1:
        df44 = df44.loc[df44.Prediction=="Exosomal"]
    else:
        df44 = df44
    df44 = round(df44,3)
    df44.to_csv(result_filename, index=None)
    for jj in glob.glob(os.getcwd()+'/'+filename+'*'):
        os.remove(jj)
    os.remove(filename_1)
    shutil.rmtree(os.getcwd()+'/fasta')
    shutil.rmtree(os.getcwd()+'/pssm_raw')
    print("\n=========Process Completed. Have an awesome day ahead.=============\n")    
#===================== Design Model Start from Here ======================
elif Model == 2:
    print('\n======= Thanks for using Predict module of ExoProPred using hybrid model. Your results will be stored in file :',result_filename,' =====\n')
    df_2,df1 = readseq(Sequence)
    lenchk(df1)
    filename = str(uuid.uuid4())
    filename_1 = str(uuid.uuid4())
    df1.to_csv(filename_1,index=None,header=False)
    feat = feature_gen(filename_1)
    df11 = pd.concat([df_2,df1],axis=1)
    df11.to_csv(filename,index=None,header=False,sep="\n")
    file_split(Sequence)
    pssm_gen(os.getcwd(),psiblast,swiss)
    os.system("python3 "+os.path.dirname(__file__)+"src/possum.py -i "+filename+" -o "+filename+"_example_aac_pssm_no_header -t aac_pssm -p "+os.getcwd()+"/pssm_raw -a 0 -b 0");
    os.system("python3 "+os.path.dirname(__file__)+"src/headerHandler.py -i "+filename+"_example_aac_pssm_no_header -p aac_pssm -n 20 -o "+filename+"_aac_pssm.csv");
    os.system("python3 "+os.path.dirname(__file__)+"src/possum.py -i "+filename+" -o "+filename+"_example_dpc_pssm_no_header -t dpc_pssm -p "+os.getcwd()+"/pssm_raw -a 5 -b 0");
    os.system("python3 "+os.path.dirname(__file__)+"src/headerHandler.py -i "+filename+"_example_dpc_pssm_no_header -p dpc_pssm -n 400 -o "+filename+"_dpc_pssm.csv");
    os.system("python3 "+os.path.dirname(__file__)+"src/possum.py -i "+filename+" -o "+filename+"_example_pssm_comp_no_header -t pssm_composition -p "+os.getcwd()+"/pssm_raw -a 5 -b 0");
    os.system("python3 "+os.path.dirname(__file__)+"src/headerHandler.py -i "+filename+"_example_pssm_comp_no_header -p pssm_composition -n 400 -o "+filename+"_pssm_comp.csv");
    X = top_feat(feat,filename+"_aac_pssm.csv",filename+"_dpc_pssm.csv",filename+"_pssm_comp.csv")
    mlres = model_run(X,'model/RF_model.pkl')
    mlres = mlres.round(3)
    os.system("perl "+merci+" -p "+filename+" -c KOOLMAN-ROHM -gl 2 -i "+motifs+"/pos_motifs.txt -o "+filename+"_pos_hits.txt")
    os.system("perl "+merci+" -p "+filename+" -c KOOLMAN-ROHM -gl 2 -i "+motifs+"/neg_motifs.txt -o "+filename+"_neg_hits.txt")
    df33 = merci_processor(filename,mlres,filename+'_pos_hits.txt',filename+'_neg_hits.txt',Threshold)
    df44 = pd.concat([df_2,df33],axis=1)
    df44.columns = ['Seq_ID','ML_Score','Positive_Motifs_Score','Negative_Motifs_Score','Total_Score','Prediction']
    if dplay == 1:
        df44 = df44.loc[df44.Prediction=="Exosomal"]
    else:
        df44 = df44
    df44 = round(df44,3)
    df44.to_csv(result_filename, index=None)
    for jj in glob.glob(os.getcwd()+'/'+filename+'*'):
        os.remove(jj)
    os.remove(filename_1)
    shutil.rmtree(os.getcwd()+'/fasta')
    shutil.rmtree(os.getcwd()+'/pssm_raw')
    print("\n=========Process Completed. Have an awesome day ahead.=============\n")
print('\n======= Thanks for using ExoProPred. Your results are stored in file :',result_filename,' =====\n\n')
