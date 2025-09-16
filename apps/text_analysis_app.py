import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    st.warning("TextBlob not available. Some features may be limited.")

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("WordCloud not available. Word cloud feature will be disabled.")

def run():
    st.title("📝 Text Analysis App")
    st.markdown("---")
    
    # Text input options
    st.sidebar.header("📋 Text Input")
    input_method = st.sidebar.selectbox(
        "Choose Input Method:",
        ["Type Text", "Upload File", "Sample Text"]
    )
    
    text = ""
    
    if input_method == "Type Text":
        text = st.text_area("Enter your text here:", height=200, placeholder="Type or paste your text here...")
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'md'])
        if uploaded_file is not None:
            try:
                text = str(uploaded_file.read(), "utf-8")
                st.success(f"✅ File uploaded successfully! ({len(text)} characters)")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    elif input_method == "Sample Text":
        sample_texts = {
            "Shakespeare Quote": "To be or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and by opposing end them.",
            "Tech Article": "Artificial intelligence and machine learning are revolutionizing the way we work and live. These technologies are being applied across various industries, from healthcare to finance, creating new opportunities and challenges. The future of AI looks promising, but we must also consider the ethical implications of these powerful tools.",
            "Product Review": "This smartphone is absolutely amazing! The camera quality is outstanding, the battery life is excellent, and the performance is smooth. However, the price is quite high, and the design could be more innovative. Overall, I would definitely recommend this product to anyone looking for a premium device."
        }
        
        selected_sample = st.selectbox("Choose a sample text:", list(sample_texts.keys()))
        text = sample_texts[selected_sample]
        st.text_area("Sample text:", value=text, height=150, disabled=True)
    
    if text.strip():
        # Text statistics
        st.subheader("📊 Text Statistics")
        
        # Basic statistics
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Characters", len(text))
        with col2:
            st.metric("Words", len(words))
        with col3:
            st.metric("Sentences", len(sentences))
        with col4:
            st.metric("Paragraphs", len(paragraphs))
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            st.metric("Avg Word Length", f"{avg_word_length:.1f}")
        
        with col2:
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            st.metric("Avg Words/Sentence", f"{avg_sentence_length:.1f}")
        
        with col3:
            unique_words = len(set(words))
            st.metric("Unique Words", unique_words)
        
        with col4:
            lexical_diversity = unique_words / len(words) if words else 0
            st.metric("Lexical Diversity", f"{lexical_diversity:.2f}")
        
        st.markdown("---")
        
        # Word frequency analysis
        st.subheader("🔤 Word Frequency Analysis")
        
        if words:
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            include_stop_words = st.checkbox("Include stop words in analysis")
            
            if include_stop_words:
                filtered_words = words
            else:
                filtered_words = [word for word in words if word not in stop_words]
            
            if filtered_words:
                word_freq = Counter(filtered_words)
                top_n = st.slider("Number of top words to display:", 5, 50, 15)
                
                # Display top words
                top_words = word_freq.most_common(top_n)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Top Words Table")
                    df_words = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                    st.dataframe(df_words, use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Word Frequency Chart")
                    st.bar_chart(pd.DataFrame(top_words, columns=['Word', 'Frequency']).set_index('Word'))
        
        # Word cloud
        if WORDCLOUD_AVAILABLE and words:
            st.markdown("---")
            st.subheader("☁️ Word Cloud")
            
            if st.button("Generate Word Cloud"):
                try:
                    # Create word cloud
                    wordcloud_text = ' '.join(filtered_words) if 'filtered_words' in locals() else ' '.join(words)
                    
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate(wordcloud_text)
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error generating word cloud: {e}")
        
        # Sentiment analysis
        if TEXTBLOB_AVAILABLE:
            st.markdown("---")
            st.subheader("😊 Sentiment Analysis")
            
            try:
                blob = TextBlob(text)
                sentiment = blob.sentiment
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Polarity", f"{sentiment.polarity:.3f}")
                    if sentiment.polarity > 0.1:
                        st.success("😊 Positive sentiment")
                    elif sentiment.polarity < -0.1:
                        st.error("😞 Negative sentiment")
                    else:
                        st.info("😐 Neutral sentiment")
                
                with col2:
                    st.metric("Subjectivity", f"{sentiment.subjectivity:.3f}")
                    if sentiment.subjectivity > 0.5:
                        st.info("📝 Subjective (opinion-based)")
                    else:
                        st.info("📰 Objective (fact-based)")
                
                # Sentiment explanation
                st.markdown("""
                **Explanation:**
                - **Polarity** ranges from -1 (very negative) to 1 (very positive)
                - **Subjectivity** ranges from 0 (objective) to 1 (subjective)
                """)
                
            except Exception as e:
                st.error(f"❌ Error in sentiment analysis: {e}")
        
        # Text preprocessing
        st.markdown("---")
        st.subheader("🔧 Text Preprocessing")
        
        if st.checkbox("Show preprocessing options"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Original text preview
                st.markdown("### Original Text (first 500 chars)")
                st.text(text[:500] + "..." if len(text) > 500 else text)
            
            with col2:
                # Preprocessing options
                lowercase = st.checkbox("Convert to lowercase", value=True)
                remove_punctuation = st.checkbox("Remove punctuation", value=True)
                remove_numbers = st.checkbox("Remove numbers", value=False)
                remove_extra_spaces = st.checkbox("Remove extra spaces", value=True)
                
                # Apply preprocessing
                processed_text = text
                
                if lowercase:
                    processed_text = processed_text.lower()
                
                if remove_punctuation:
                    processed_text = re.sub(r'[^\w\s]', '', processed_text)
                
                if remove_numbers:
                    processed_text = re.sub(r'\d+', '', processed_text)
                
                if remove_extra_spaces:
                    processed_text = ' '.join(processed_text.split())
                
                st.markdown("### Processed Text (first 500 chars)")
                st.text(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)
        
        # Text patterns
        st.markdown("---")
        st.subheader("🔍 Text Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Email detection
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            st.metric("Email Addresses", len(emails))
            if emails:
                for email in emails[:5]:  # Show first 5
                    st.text(f"📧 {email}")
        
        with col2:
            # URL detection
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            st.metric("URLs", len(urls))
            if urls:
                for url in urls[:5]:  # Show first 5
                    st.text(f"🔗 {url}")
        
        # Character frequency
        if st.checkbox("Show character frequency analysis"):
            st.markdown("### 🔤 Character Frequency")
            char_freq = Counter(text.lower())
            # Remove spaces and newlines for cleaner display
            char_freq = {char: freq for char, freq in char_freq.items() if char.isalnum()}
            
            if char_freq:
                top_chars = dict(sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:15])
                df_chars = pd.DataFrame(list(top_chars.items()), columns=['Character', 'Frequency'])
                st.bar_chart(df_chars.set_index('Character'))
    
    else:
        st.info("👆 Please enter some text to analyze!")
        
        # Show available features
        st.markdown("""
        ## 🔧 Available Features:
        
        - **Text Statistics**: Character, word, sentence, and paragraph counts
        - **Word Frequency Analysis**: Most common words with filtering options
        - **Word Cloud**: Visual representation of word frequency
        - **Sentiment Analysis**: Polarity and subjectivity scores
        - **Text Preprocessing**: Clean and normalize text
        - **Pattern Detection**: Find emails, URLs, and other patterns
        - **Character Analysis**: Character frequency distribution
        """)

if __name__ == "__main__":
    run()