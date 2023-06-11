#====================================================================================
#                                                                                   #
#                                                                                   #
#                               UTILITIES                                           #
#                                                                                   #
#                                                                                   #
#====================================================================================




import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import json
from sys import argv
import time
nltk.download('punkt')




#====================================================================================
#                                                                                   #
#                                                                                   #
#                               GLOBAL VARIABLES                                    #
#                                                                                   #
#                                                                                   #
#====================================================================================





#load stopwords to eliminate them while processing
file = open("Stopword-List.txt","r")
content = file.read()
file.close()
stopWords = content.split()





#====================================================================================
#                                                                                   #
#                                                                                   #
#                               FUNCTIONS                                           #
#                                                                                   #
#                                                                                   #
#====================================================================================






def stopWordsRemoval(tokens):
    
    #return all tokens which are not part of stopWords
    words = []
    for token in tokens:
        if token not in stopWords:
            words.append(token)
    
    return words





def case_fold(token):
    #convert token to lower case to normalize it
    token = token.lower()

    word = token
    token = ""

    #only conisder english alphabets in the token, not special characters
    for letter in word:
        if letter >= 'a' and letter <= 'z':
            token = token + letter
    
    return token





def stem(token):
    
    #use porter stemmer to stem the token
    stemmer = PorterStemmer()
    stemmedWord = stemmer.stem(token)

    return stemmedWord




def normalize(token):
    #apply case folding
    token = case_fold(token)
    #apply stemming
    token = stem(token)
    return token





def tokenizer(content):
    #extract different words from the string
    content = content.replace("-", " ")
    content = content.replace("•", " ")
    words = word_tokenize(content)

    return words





def parseDoc(content):

    words = tokenizer(content)

    tokens = []
    for word in words:

        #apply normalization to token
        token = normalize(word)

        #make sure there are no empty spaces in the tokens list
        if len(token) > 0:
            tokens.append(token)
    
    #remove stopwords from list
    tokens = stopWordsRemoval(tokens)

    return tokens




def parseQuery(query):
    #tokenize the query and return tokens
    tokens = tokenizer(query)

    return tokens




def saveInvertedIndex(invertedIndex):
    #write the index to json file
    with open("InvertedIndex.json","w") as file:
        json.dump(invertedIndex,file)




def savePosIndex(posIndex):
    #write the index to json file
    with open("PosIndex.json","w") as file:
        json.dump(posIndex,file)




def loadInvertedIndex():
    #load index from json file and return
    with open("InvertedIndex.json","r") as file:
        invertedIndex = json.load(file)
    return invertedIndex




def loadPosIndex():
    #load index from json file and return
    with open("PosIndex.json","r") as file:
        posIndex = json.load(file)
    return posIndex




def fetchPostingList(term, invertedIndex):

    #return postings list from the inverted index for that term
    return invertedIndex[term][1]




def fetchPositions(term,doc, posIndex):

    #return positions list for that term from that doc
    return posIndex[doc-1][term]



#====================================================================================
#                                                                                   #
#                                                                                   #
#                               DOCUMENT PROCESSING                                 #
#                                                                                   #
#                                                                                   #
#====================================================================================




def processDocs():
    posIndex = []   #used to store positional indices
    globalDict = [] #keep track of all terms used in all docs

    for i in range(1,31,1):

        #open doc and read it
        docName = f"{i}.txt"
        doc = open("Dataset/"+docName,"r")
        docContent = doc.read()
        doc.close()

        #parse document and get tokens
        tokens = parseDoc(docContent)

        #generate positional index for current doc
        posIndex.append({})

        for pos in range(len(tokens)):

            token = tokens[pos]

            #append the position inside the doc of the token to the positional Index
            if token not in posIndex[i-1].keys():
                posIndex[i-1][token] = [pos]
            else:
                posIndex[i-1][token].append(pos)
            
            #keep track of all unique tokens
            if token not in globalDict:
                globalDict.append(token)

    #sort all terms in alphabetical order
    globalDict.sort()

    invertedIndex = {}

    #initialise each words posting list as an empty list
    for word in globalDict:
        invertedIndex[word] = (0,[])

    #for each term, is it appears in some document, append the doc ID in its postings list
    for term in invertedIndex.keys():
        doc_freq = 0
        
        for docID in range(1,31,1):
            if term in posIndex[docID-1].keys():
                #if term exists in some doc, increment the doc freq
                doc_freq += 1
                invertedIndex[term][1].append(docID)
        
        invertedIndex[term] = (doc_freq,invertedIndex[term][1])
    
    #save index
    saveInvertedIndex(invertedIndex)
    savePosIndex(posIndex)





#====================================================================================
#                                                                                   #
#                                                                                   #
#                               QUERY PROCESSING                                    #
#                                                                                   #
#                                                                                   #
#====================================================================================





def proximityQuery(tokens, invertedIndex):

    #load positional index
    posIndex = loadPosIndex()

    #get k, the maximum number of words between both terms
    k = int(tokens[3])

    #extract terms from query and normalize them
    terms = []
    for i in range(0,2,1):
        terms.append(normalize(tokens[i]))

    result = set()

    #iterate through every doc in posting list of 1st term
    for doc in fetchPostingList(terms[0], invertedIndex):
        
        #make sure 2nd term also exists in that doc
        if doc not in fetchPostingList(terms[1], invertedIndex):
            continue
        
        #get positions list of both terms in that doc
        list1 = fetchPositions(terms[0],doc,posIndex)
        list2 = fetchPositions(terms[1],doc,posIndex)
        
        #iterators for list
        l1 = 0
        l2 = 0

        #iterate over both lists
        while l1 < len(list1) and l2 < len(list2):
            #check number of words between positions of both words
            diff = list1[l1] - list2[l2]

            #if the difference is within the acceptable range, append it to result
            if diff <=k and diff >=-k:
                result.add(doc)
                break
            #otherwise increment the iterator
            else:
                if diff > k:
                    l2+=1
                else:
                    l1+=1

    return result




def booleanQuery(query, invertedIndex):
    
    #calculate length of query
    queryLength = len(query)
    result = set()
    
    op_list = ["AND","and","OR","or","NOT","not"]
    
    #traverse the query
    for i in range(0,queryLength,1):

        #if an operator is encountered, pass the rest of the query to processQuery function
        #the operation will be performed on its result
        if query[i] in op_list:
            
            #store which operation is to be performed
            operator = query[i]
            
            #join the tokens and make them a space separated string to pass a valid string query
            newQuery = " ".join(query[i+1:])
            
            #get result of the rest of the query and store it
            result = processQuery(newQuery)[0]
            break
        
        #if a term is encountered, normalize it and store it to use later
        else:
            term = normalize(query[i])
    
    #if it's a single word query, result will be it's postings list
    if queryLength == 1:
        if term in invertedIndex.keys():
            result = set(invertedIndex[term][1])
        return result
    
    #for AND operator
    if operator == "AND" or operator == "and":

        #if term doesn't exist in vocabulary, result will be empty
        if term not in invertedIndex.keys():
            result = set()
        
        #otherwise perform intersection of the term's posting list and the result of the rest of the query
        else:
            result = result.intersection(invertedIndex[term][1])
    
    elif operator == "OR" or operator == "or":
        
        #if term exists in vocabulary, add its postings list to result
        if term in invertedIndex.keys():
            result = result.union(invertedIndex[term][1])
    
    elif operator == "NOT" or operator == "not":

        #result will contain all such docs which do not contain the docs we received by processing the rest of the query
        tempResult = set()
        for doc in range(1,31,1):
            if doc not in result:
                tempResult.add(doc)
        result = tempResult
    
    return result
    





def processQuery(query):
    # Record the start time
    start_time = time.time()

    #load inverted index for processing as it is needed in processing
    invertedIndex = loadInvertedIndex()

    #Parse query
    query = parseQuery(query)
    

    #if query has more than 2 tokens and contains '/', it must be a proximity query
    #otherwise it's a boolean query
    if len(query) > 2:
        if query[2] == '/':
            result = proximityQuery(query, invertedIndex)
        else:
            result = booleanQuery(query,invertedIndex)
    else:
        result = booleanQuery(query,invertedIndex)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the time elapsed
    time_taken = end_time - start_time
    
    return result, time_taken