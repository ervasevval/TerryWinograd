from book_analysis import BookAnalysis

def main():
    # Kitap JSON dosyasının yolunu belirle
    file_path = r'C:\Users\w\Desktop\Kodlama\VsCode\HelloWorld.py\kitap.json'  # Burada kitaptaki JSON dosyasının yolunu belirtin

    # Kitap analizi sınıfını başlat
    book_analysis = BookAnalysis(file_path)

    # Kitap hakkında temel bilgileri keşfet
    book_info = book_analysis.explore_data()
    print(f"Book Title: {book_info['title']}")
    print(f"Author: {book_info['author']}")
    print(f"Total Sections: {book_info['sections']}")
    print(f"Total Text Length: {book_info['text_length']} characters")

    # Kelime, cümle, paragraf sayısını öğrenme
    word_count = book_analysis.count_words()
    sentence_count = book_analysis.count_sentences()
    paragraph_count = book_analysis.count_paragraphs()

    print(f"Word Count: {word_count}")
    print(f"Sentence Count: {sentence_count}")
    print(f"Paragraph Count: {paragraph_count}")

# Ana fonksiyonu çalıştır
if __name__ == '__main__':
    main()