import streamlit as st
#########################" Config Page"
st.set_page_config(
    page_title="Discussion",
    page_icon="static/icon.png",  
)
#########################################
st.markdown(""" ### 📋 Discussion """)
st.markdown(
    """
    To ensure the models maintain their performance even in the face of potential changes in data patterns, it's advisable to implement continuous learning strategies. While the models currently yield good results and demonstrate consistent performance on new data, there remains the possibility of significant shifts in data patterns, particularly over extended periods. Such changes could adversely impact the models' performance.
    
    To address this concern and preserve the models' ability to generalize effectively, I recommend adopting a continuous learning approach. This involves periodically retraining the models on new data, thereby incorporating the latest information into their knowledge base. It's important to recognize that future data patterns may undergo significant changes at varying speeds, necessitating ongoing monitoring and adaptation of the models to maintain their efficacy.    
    """
)
st.markdown("<h3>It was a pleasure to collaborate with you Mr.Tom in this insightful project. Thanks for trusting me and Have a good luck! 💯✔️🙏</h3>", unsafe_allow_html=True)
