import sys
import argparse
import os
import json
import re
import spacy
import html
from itertools import chain

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')

def preprocess(comment):
    """ 
    This function preprocesses a single comment.

    Parameters:                                                                      
    - comment: string, the body of a comment.

    Returns:
    - modified_comment: string, the modified comment.
    """
    modified_comment = comment

    # Replace newlines with spaces to handle other whitespace chars.
    modified_comment = re.sub(r"\s{1,}", " ", modified_comment)

    # Remove '[deleted]' or '[removed]' statements.
    modified_comment = re.sub(r"\[deleted\]|\[removed\]", "", modified_comment)

    # Unescape HTML.
    modified_comment = html.unescape(modified_comment)
    # Remove URLs.
    modified_comment = re.sub(r"(http|www)\S+", "", modified_comment)

    # Remove duplicate spaces.
    modified_comment = re.sub(r"\s{2,}", " ", modified_comment)

    spacy_doc = nlp(modified_comment)
    
    # Maintain a list of sentences where each sentence is a list of tagged tokens
    tagged_sents = []
    
    for sent in spacy_doc.sents:
        
        # Maintain a list of tagged tokens for current sentence
        tagged_tokens = []
        
        for token in sent:

            lemma = token.lemma_
            if token.lemma_.startswith("-") and not token.text.startswith("-"):
                lemma = token.text

            if token.text.isupper():
                lemma = lemma.upper()
            else:
                lemma = lemma.lower()

            tagged_tokens.append(lemma + "/" + token.tag_)
        
        tagged_sents.append(tagged_tokens)
    
    # Add space between tokens and newline character between sentences
    tagged_sents = [" ".join(sent) for sent in tagged_sents]
    modified_comment = "\n".join(tagged_sents)

    return modified_comment


def main(args):
    all_output = []
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))
            
            for i in range(args.max):
                j = json.loads(data[i])

                all_output.append({
                    # Required fields
                    "id": j["id"],
                    "cat": file,
                    "body": preprocess(j["body"]),
                    
                    # Temporary fields
                    "parent_id": j["parent_id"],
                    "original_body": j["body"],
                })
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(all_output))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Specify the output file.", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file.", default=10000)
    parser.add_argument("--data-dir", help="Reddit data directory with JSON files for each category.", required=True)
    
    args = parser.parse_args()
    main(args)
