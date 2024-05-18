from utils import extract_bigrams, read_text_file, summarize_text, extract_keywords_using_KeyBert

def main():
    # Chemin du fichier texte d'entrée
    file_path = "/content/drive/MyDrive/topic/article1.txt"
    
    # Lire le contenu du fichier texte
    text = read_text_file(file_path)
    
    # Extraire les bigrammes du texte
    bigrams = extract_bigrams(text)
    
    # Enregistrer les bigrammes dans un fichier texte
    with open("/content/drive/MyDrive/topic/bigrams.txt", "w", encoding="utf-8") as f:
        for bigram in bigrams:
            f.write("{}\n".format(bigram))
    
    # Résumer le texte
    summary = summarize_text(file_path)
    
    # Enregistrer le résumé dans un fichier texte
    with open("/content/drive/MyDrive/topic/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    #Extraire les ngram en utilisant Keybert
    keywords = extract_keywords_using_KeyBert(file_path, top_n=80)
    print(keywords)

if __name__ == "__main__":
    main()



