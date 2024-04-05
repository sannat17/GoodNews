import numpy as np
import argparse
import json
import csv
import os
import re

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

PUNCTUATIONS = {
    # Punctuations in en_core_web_sm tagger:
    "$", "''", "``", ",", ".", ":",

    # Not in en_core_web_sm tagger but added because of handout:
    "'", "`",
    "#", ":", "(", ")"
}

CLASS_FEAT = {
    "Left": 0,
    "Center": 1,
    "Right": 2,
    "Alt": 3
}

def extract1(comment):
    """ 
    This function extracts features from a single comment.

    Parameters:
    - comment: string, the body of a comment (after preprocessing).

    Returns:
    - feats: NumPy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here).
    """    
    feats = np.zeros(173)

    # Common structures for various features:
    sent_lengths = [len(s.split()) for s in comment.split("\n")]
    bgl_aoa_vals = []
    bgl_img_vals = []
    bgl_fam_vals = []
    warr_v_mean_vals = []
    warr_a_mean_vals = []
    warr_d_mean_vals = []

    # Trackers for different features handled in for loop:
    go_vbg_before = False # 7.
    token_count = 0 # 16.
    char_count = 0 # 16.

    for t in comment.split():

        parts = t.split("/")
        tag = parts[-1]
        # Handle case where lemma may also contain "/"
        lemma = "/".join(parts[:-1])
        
        # 1. Number of tokens in upper case excluding tag (>= 3 letters long)
        if lemma.isupper() and len([ch for ch in lemma if ch.isalpha()]) >= 3:
            feats[0] += 1

        # 2. Number of first-person pronouns.
        if lemma.lower() in FIRST_PERSON_PRONOUNS:
            feats[1] += 1

        # 3. Number of second-person pronouns.
        if lemma.lower() in SECOND_PERSON_PRONOUNS:
            feats[2] += 1

        # 4. Number of third-person pronouns.
        if lemma.lower() in THIRD_PERSON_PRONOUNS:
            feats[3] += 1

        # 5. Number of coordinating conjunctions.
        if tag == "CC":
            feats[4] += 1

        # 6. Number of past-tense verbs.
        if tag == "VBD":
            feats[5] += 1

        # 7. Number of future-tense verbs.
        if tag == "MD" and lemma.lower() in ["will"]:
            feats[6] += 1
        if go_vbg_before and tag == "TO":
                feats[6] += 1
        if lemma.lower() == "go" and tag == "VBG":
            go_vbg_before = True
        else:
            go_vbg_before = False        

        # 8. Number of commas.
        if tag == ",":
            feats[7] += 1

        # 9. Number of multi-character punctuation tokens.
        if len(lemma) > 1 and tag in PUNCTUATIONS:
            feats[8] += 1
        
        # 10. Number of common nouns.
        if tag in {"NN", "NNS"}:
            feats[9] += 1

        # 11. Number of proper nouns.
        if tag in {"NNP", "NNPS"}:
            feats[10] += 1

        # 12. Number of adverbs.
        if tag in {"RB", "RBR", "RBS"}:
            feats[11] += 1

        # 13. Number of wh- words.
        if tag in {"WDT", "WP", "WP$", "WRB"}:
            feats[12] += 1

        # 14. Number of slang acronyms.
        if lemma.lower() in SLANG:
            feats[13] += 1

        # Dealing with common and feature specific structures:
        
        # 16. Average length of tokens, excluding punctuation-only tokens, in characters.
        if lemma not in PUNCTUATIONS:
            token_count += 1
            char_count += len(lemma)
        
        # Bristol, Gilhooly, and Logie norms
        if lemma.lower() in BGL_NORMS:
            row = BGL_NORMS[lemma.lower()]
            aoa = row["AoA (100-700)"]
            img = row["IMG"]
            fam = row["FAM"]
            if aoa.replace(".", "").isdigit():
                bgl_aoa_vals.append(float(aoa))
            if img.replace(".", "").isdigit():
                bgl_img_vals.append(float(img))
            if fam.replace(".", "").isdigit():
                bgl_fam_vals.append(float(fam))

        # Warriner norms
        if lemma.lower() in WARR_NORMS:
            row = WARR_NORMS[lemma.lower()]
            v_mean = row["V.Mean.Sum"]
            a_mean = row["A.Mean.Sum"]
            d_mean = row["D.Mean.Sum"]
            if v_mean.replace(".", "").isdigit():
                warr_v_mean_vals.append(float(v_mean))
            if a_mean.replace(".", "").isdigit():
                warr_a_mean_vals.append(float(a_mean))
            if d_mean.replace(".", "").isdigit():
                warr_d_mean_vals.append(float(d_mean))


    # 15. Average length of sentences, in tokens.
    feats[14] = sum(sent_lengths) / len(sent_lengths)

    # 16. Average length of tokens, excluding punctuation-only tokens, in characters.
    feats[15] = char_count / token_count if token_count > 0 else 0

    # 17. Number of sentences..
    feats[16] = len(sent_lengths)

    # Helper function to find mean
    find_mean = lambda x: float(np.mean(x)) if len(x) > 0 else 0
    # Helper function to find standard deviation
    find_std = lambda x: float(np.std(x)) if len(x) > 0 else 0

    # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms.
    feats[17] = find_mean(bgl_aoa_vals)
    # 19. Average of IMG from Bristol, Gilhooly, and Logie norms.
    feats[18] = find_mean(bgl_img_vals)
    # 20. Average of FAM from Bristol, Gilhooly, and Logie norms.
    feats[19] = find_mean(bgl_fam_vals)

    # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms.
    feats[20] = find_std(bgl_aoa_vals)
    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms.
    feats[21] = find_std(bgl_img_vals)
    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms.
    feats[22] = find_std(bgl_fam_vals)

    # 24. Average of V.Mean.Sum from Warringer norms.
    feats[23] = find_mean(warr_v_mean_vals)
    # 25. Average of A.Mean.Sum from Warringer norms.
    feats[24] = find_mean(warr_a_mean_vals)
    # 26. Average of D.Mean.Sum from Warringer norms.
    feats[25] = find_mean(warr_d_mean_vals)

    # 27. Standard deviation of V.Mean.Sum from Warringer norms.
    feats[26] = find_std(warr_v_mean_vals)
    # 28. Standard deviation of A.Mean.Sum from Warringer norms.
    feats[27] = find_std(warr_a_mean_vals)
    # 29. Standard deviation of D.Mean.Sum from Warringer norms.
    feats[28] = find_std(warr_d_mean_vals)

    return feats
    
    
def extract2(feats, comment_class, comment_id):
    """ This function adds features 30-173 for a single comment.

    Parameters:
    - feats: np.array of length 173.
    - comment_class: str in {"Alt", "Center", "Left", "Right"}.
    - comment_id: int indicating the id of a comment.

    Returns:
    - feats: NumPy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    """
    if comment_id in LIWC_FEATS[comment_class]["ids"]:
        feats[29:] = LIWC_FEATS[comment_class]["feats"][LIWC_FEATS[comment_class]["ids"][comment_id]]
    else:
        feats[29:] = np.zeros(144)

    return feats


def main(args):
    # Declare necessary global variables here. 
    global BGL_NORMS, WARR_NORMS, LIWC_FEATS

    # Load data
    data = json.load(open(args.input))

    # Load BGL wordlist
    with open(args.bgl_path, "r") as f:
        BGL_NORMS = {}
        reader = csv.DictReader(f)
        for row in reader:
            BGL_NORMS[row["WORD"]] = row

    # Load Warriner wordlist
    with open(args.warr_path, "r") as f:
        WARR_NORMS = {}
        reader = csv.DictReader(f)
        for row in reader:
            WARR_NORMS[row["Word"]] = row
    
    # Load LIWC features for each category
    LIWC_FEATS = {}
    for cat in ["Alt", "Center", "Left", "Right"]:
        comment_ids_file = os.path.join(args.feats_dir, f"{cat}_IDs.txt")
        feats_file = os.path.join(args.feats_dir, f"{cat}_feats.dat.npy")
        LIWC_FEATS[cat] = {
            "ids": dict(),
            "feats": np.load(feats_file)
        }
        with open(comment_ids_file, "r") as f:
            for i, comment_id in enumerate(f.read().split("\n")):
                LIWC_FEATS[cat]["ids"][comment_id] = i

    # Extract features and make dataset numpy array
    feats = np.zeros((len(data), 173+1))
    for i, d in enumerate(data):
        curr = extract2(extract1(d["body"]), d["cat"], d["id"])
        feats[i,:-1] = curr
        feats[i,-1] = CLASS_FEAT[d["cat"]]

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Specify the output file.", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1.", required=True)
    parser.add_argument("--feats-dir", help="The directory containing the LIWC features.", required=True)
    parser.add_argument("--bgl-path", help="The path to the Bristol, Gilhooly, and Logie norms csv", required=True)
    parser.add_argument("--warr-path", help="The path to the Warringer norms csv", required=True)
    args = parser.parse_args()

    main(args)

