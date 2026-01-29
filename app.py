import streamlit as st
from groq import Groq
import re
import os
from typing import Optional

# Your Groq API Key - Replace with your actual API key or use environment variable
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE"))

# Working model name
WORKING_MODEL = "llama-3.1-8b-instant"  # Current Groq model

# Configure page
st.set_page_config(
    page_title="Postify - AI Social Media Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .post-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .hashtag {
        color: #1da1f2;
        font-weight: bold;
    }
    .error-message {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
        color: #cc0000;
    }
    .success-message {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #44ff44;
        color: #006600;
    }
</style>
""", unsafe_allow_html=True)

def configure_groq() -> bool:
    """Configure Groq API with the hardcoded key"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Test the API key by making a simple request
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model=WORKING_MODEL,
            max_tokens=10
        )
        
        return True
        
    except Exception as e:
        print(f"Exception during Groq API test: {str(e)}")
        return False

def generate_social_media_post(blog_content: str, platform: str, tone: str) -> Optional[str]:
    """Generate social media post using Groq API"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Create platform-specific prompts
        platform_specs = {
            "Twitter": "MAXIMUM 250 characters including hashtags, punchy and engaging",
            "LinkedIn": "professional tone, 1-3 paragraphs, business-focused",
            "Facebook": "casual and friendly, 1-2 paragraphs with emojis",
            "Instagram": "visual-focused with emojis and engaging caption"
        }
        
        # Special handling for Twitter to ensure character limit
        if platform == "Twitter":
            prompt = f"""
            Create a Twitter post from this blog article. CRITICAL: The entire post must be under 250 characters including hashtags.
            
            Requirements:
            - Maximum 250 characters total (including spaces and hashtags)
            - Extract 1-2 key points only
            - Make it {tone} in tone
            - Add 2-3 relevant hashtags maximum
            - Create urgency to read the full article
            - Be punchy and engaging
            
            Blog Content:
            {blog_content}
            
            Generate ONLY the Twitter post. Keep it short and impactful.
            """
        else:
            prompt = f"""
            Convert the following blog article into an engaging {platform} post that will make people want to click and read the full article.
            
            Platform: {platform} ({platform_specs.get(platform, "engaging social media post")})
            Tone: {tone}
            
            Requirements:
            1. Extract the most interesting and compelling points from the blog
            2. Create curiosity and urgency to read the full article
            3. Include relevant hashtags at the end (5-10 hashtags)
            4. Make it {tone} in tone
            5. Ensure it's optimized for {platform}
            
            Blog Content:
            {blog_content}
            
            Generate only the social media post content with hashtags. Do not include any explanations or additional text.
            """
        
        # Make API request to Groq
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=WORKING_MODEL,
            max_tokens=1024,
            temperature=0.7
        )
        
        generated_content = response.choices[0].message.content
        
        # Special handling for Twitter - truncate if still too long
        if platform == "Twitter" and len(generated_content) > 280:
            # Try to truncate intelligently
            lines = generated_content.split('\n')
            main_content = lines[0] if lines else generated_content
            
            # Extract hashtags
            hashtags = re.findall(r'#\w+', generated_content)
            hashtag_text = ' ' + ' '.join(hashtags[:2]) if hashtags else ''
            
            # Calculate available space for main content
            available_chars = 280 - len(hashtag_text)
            
            if len(main_content) > available_chars:
                # Truncate main content and add ellipsis
                main_content = main_content[:available_chars-4] + "..."
            
            generated_content = main_content + hashtag_text
        
        return generated_content
        
    except Exception as e:
        st.error(f"Error generating post: {str(e)}")
        return None

def extract_hashtags(post_content: str) -> list:
    """Extract hashtags from the generated post"""
    hashtags = re.findall(r'#\w+', post_content)
    return hashtags

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Postify</h1>', unsafe_allow_html=True)
    st.markdown("**Transform your blog articles into viral social media posts with AI**")
    st.markdown("*Powered by Groq AI - Generate engaging content for Twitter, LinkedIn, Facebook & Instagram*")
    
    st.divider()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show API key status
        if GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
            st.success("✅ Groq API Key is set in code")
            # Optional: Test the API key (but don't block if it fails)
            try:
                if configure_groq():
                    st.success("🔗 Groq API connection test passed")
                else:
                    st.warning("⚠️ API test failed, but you can still try generating posts")
            except:
                st.info("ℹ️ Skipping API test - will validate during generation")
        else:
            st.warning("⚠️ Please set your Groq API key in the code (GROQ_API_KEY variable)")
        
        st.divider()
        
        # Platform selection
        platform = st.selectbox(
            "📱 Target Platform",
            ["Twitter", "LinkedIn", "Facebook", "Instagram"],
            help="Choose the social media platform for optimization"
        )
        
        # Tone selection
        tone = st.selectbox(
            "🎭 Tone",
            ["Professional", "Casual", "Exciting", "Informative", "Humorous"],
            help="Select the tone for your social media post"
        )
        
        st.divider()
        
        # Tips
        st.header("💡 Tips")
        st.markdown("""
        - **Paste your full blog article** for best results
        - **AI extracts key points** automatically  
        - **Platform-optimized** content generation
        - **Smart hashtag** suggestions included
        - **Character limits** respected for each platform
        """)
        
        st.divider()
        
        st.markdown("**🎯 Perfect for:**")
        st.markdown("""
        - Content creators & bloggers
        - Social media managers  
        - Digital marketers
        - Business owners
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Blog Content")
        
        # Blog URL input (optional)
        blog_url = st.text_input(
            "Blog URL (optional)",
            placeholder="https://example.com/blog-post",
            help="Paste the blog URL for reference"
        )
        
        # Blog content input
        blog_content = st.text_area(
            "Paste your blog article here",
            height=400,
            placeholder="Paste the full blog article content here...",
            help="Include title, main content, and key points"
        )
        
        # Generate button
        generate_button = st.button(
            "🚀 Generate Social Media Post",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.header("📱 Generated Social Media Post")
        
        if generate_button:
            if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
                st.markdown('<div class="error-message">❌ Please set your Groq API key in the code (GROQ_API_KEY variable)</div>', unsafe_allow_html=True)
            elif not blog_content.strip():
                st.markdown('<div class="error-message">❌ Please paste your blog content</div>', unsafe_allow_html=True)
            else:
                with st.spinner(f"Generating {platform} post..."):
                    generated_post = generate_social_media_post(blog_content, platform, tone)
                    
                    if generated_post:
                        st.markdown('<div class="success-message">✅ Post generated successfully!</div>', unsafe_allow_html=True)
                        
                        # Display the generated post
                        st.markdown('<div class="post-container">', unsafe_allow_html=True)
                        st.markdown(f"**Platform:** {platform}")
                        st.markdown(f"**Tone:** {tone}")
                        st.divider()
                        st.markdown(generated_post)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Extract and display hashtags
                        hashtags = extract_hashtags(generated_post)
                        if hashtags:
                            st.markdown("**Hashtags used:**")
                            hashtag_display = " ".join([f'<span class="hashtag">{tag}</span>' for tag in hashtags])
                            st.markdown(hashtag_display, unsafe_allow_html=True)
                        
                        # Copy to clipboard button
                        st.text_area(
                            "Copy the post:",
                            value=generated_post,
                            height=150,
                            help="Select all and copy to use in your social media platform"
                        )
                        
                        # Character count for Twitter
                        if platform == "Twitter":
                            char_count = len(generated_post)
                            if char_count <= 280:
                                st.success(f"✅ Character count: {char_count}/280")
                            else:
                                st.warning(f"⚠️ Character count: {char_count}/280 (Too long for Twitter)")
                        
                        # Additional options
                        st.divider()
                        col_regenerate, col_download = st.columns(2)
                        
                        with col_regenerate:
                            if st.button("🔄 Regenerate", use_container_width=True):
                                st.rerun()
                        with col_download:
                            st.download_button(
                                "💾 Download Post",
                                data=generated_post,
                                file_name=f"{platform.lower()}_post.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with ❤️ using Streamlit and Groq AI</p>
        <p><strong>Postify</strong> - Turn your blogs into viral social media content!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()