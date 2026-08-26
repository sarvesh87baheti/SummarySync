from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3")

def summarize_text(text):
    try:
        if len(text) > 2048:  
            text = text[:2048]
        
        prompt = f"Summarize the following text:\n\n{text}\n\n"
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        return f"An error occurred: {e}"

text = """
Planning to invest in small cap mutual funds? For optimal protection against losses in small-cap mutual funds, investors should maintain Systematic Investment Plans (SIPs) for a minimum duration of 10-12 years, according to a new analysis. The analysis by Valuemetrics Technologies, examining monthly rolling returns for SIPs between April 2005 and March 2025 in the Nifty Small Cap 250 Total Returns Index, revealed losses in investments spanning three to ten years. According to an ET report, the study indicated that three-year SIPs achieved a maximum return of 42.1% annually, whilst experiencing significant depreciation of up to 64.70% in the worst scenarios during this period. The analysis further showed that 10-year SIP investments resulted in a 0.4% annual loss, whereas extending the investment period to 12 years yielded a minimum return of 2.4%.
"Investors randomly start SIPs in small cap funds, but it is important that they should know that they need to come with longer time horizons if they need to protect capital in this category," said Chirag Patel, Co-Founder, Valuemetrics Technologies. "As investors extend their investment horizon, the impact of short term volatility diminishes, increasing the probability of earning higher returns,” he was quoted as saying by the financial daily.
Since September 24, when smaller shares experienced their peak, the Nifty Small Cap 250 has declined by 18.4%. SIP investments in this index have decreased by 13.7%, whilst the Nifty Midcap 150 index SIP value reduced by 7.5%. The Nifty 50 staggered investments showed a modest 1% reduction. This downturn led to a 34% reduction in retail investments in mid-cap and small-cap funds in February compared to January. The research demonstrated that investors who maintained SIPs in Nifty Small Cap 250 TRI for 12 to 15 years consistently avoided losses. "A well managed small cap fund can generate higher returns than its large cap counterparts," said S Shankar, CFP, Credo Capital. "However due to high interim volatility in this space investors must have long time frames typically of 8-10 years."

"""  # Your full text here

summary = summarize_text(text)
print(summary)
