# Postify 🚀

Transform your blog articles into viral social media posts with AI! Generate engaging content for Twitter, LinkedIn, Facebook & Instagram using Groq's lightning-fast AI.

## 🌐 Live Demo

[Try Postify here!](https://your-app-url.streamlit.app) *(Update this after deployment)*

## Features ✨

- **Multi-Platform Support**: Generate posts optimized for Twitter, LinkedIn, Facebook, and Instagram
- **Smart Content Extraction**: AI automatically identifies the most interesting points from your blog
- **Customizable Tone**: Choose from Professional, Casual, Exciting, Informative, or Humorous tones
- **Auto Hashtag Generation**: Relevant hashtags are automatically added based on content
- **Character Count Tracking**: Real-time character counting for Twitter optimization
- **Copy & Download**: Easy copying and downloading of generated posts
- **Beautiful UI**: Clean, responsive Streamlit interface
- **Powered by Groq**: Lightning-fast AI generation with Llama 3.1

## Setup Instructions 🚀

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/postify.git
cd postify
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Groq API Key
1. Go to [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the API key

### 4. Set Your API Key
Edit `app.py` line 8:
```python
GROQ_API_KEY = "your_actual_groq_api_key_here"
```

### 5. Run the Application
```bash
streamlit run app.py
```

### 6. Access the App
Open your browser and go to `http://localhost:8501`

## How to Use 📖

1. **Set API Key**: Add your Groq API key in the code
2. **Select Platform**: Choose your target social media platform
3. **Choose Tone**: Select the desired tone for your post
4. **Paste Blog Content**: Copy and paste your blog article in the text area
5. **Generate**: Click "Generate Social Media Post" button
6. **Copy & Use**: Copy the generated post and use it on your social media platform

## Platform Optimizations 🎯

- **Twitter**: 280 character limit with smart truncation, punchy and engaging
- **LinkedIn**: Professional tone, business-focused, 1-3 paragraphs
- **Facebook**: Casual and friendly with emojis, 1-2 paragraphs
- **Instagram**: Visual-focused with emojis and engaging captions

## Deployment 🚀

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Set environment variables (see below)
5. Deploy the app

### Environment Variables for Production

For production deployment, set your Groq API key as an environment variable:
- Variable name: `GROQ_API_KEY`
- Value: Your actual API key

## Available Models 🤖

- `llama-3.1-8b-instant` (default - fast and efficient)
- `llama-3.1-70b-versatile` (more powerful but slower)
- `mixtral-8x7b-32768` (good alternative)

## File Structure 📁

```
postify/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore patterns
└── .streamlit/        # Streamlit config
    ├── config.toml
    └── secrets.toml
```

## Tips for Best Results 💡

- Include the full blog article with title and main content
- The more detailed your blog content, the better the AI can extract key points
- Try different tones to see what works best for your audience
- Use the regenerate feature to get alternative versions
- Check character counts for Twitter posts

## Contributing 🤝

Feel free to fork this project and submit pull requests for improvements!

## License 📄

This project is open source and available under the MIT License.

---

**Built with ❤️ using Streamlit and Groq AI**

**Postify** - Turn your blogs into viral social media content!