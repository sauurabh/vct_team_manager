import streamlit as st
import os
from pathlib import Path
import json
from team_manager import main

st.title("VALORANT Esports Team Manager")
st.subheader("Team Overview")

question = st.text_input("Enter your question about the team:")

if question:
    try:
        response = main(question)
        
        if isinstance(response, str):
            team_data = json.loads(response)
        else:
            team_data = response  
        if isinstance(team_data, dict) and "teamOverview" in team_data:
            st.write("### Team Overview")
            overview = team_data["teamOverview"]
        if isinstance(team_data, dict) and "team" in team_data:
            st.write("### All Player Names")
            player_names = [player["playerName"] for player in team_data["team"]]
            st.write(", ".join(player_names))
            
            st.write("**Potential Synergies and Strategies:**")
            if "potentialSynergiesAndStrategies" in overview:
                st.write(overview["potentialSynergiesAndStrategies"])
            
            st.write("**Preparedness for Competitive Play:**")
            if "preparednessForCompetitivePlay" in overview:
                st.write(overview["preparednessForCompetitivePlay"])

            # Display detailed player information
            st.subheader("Player Details")
            
            # Create columns for better visualization
            for player in team_data["team"]:
                with st.expander(f"{player.get('role', 'Unknown Role')}: {player.get('playerName', 'Unknown Player')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Information**")
                        st.write(f"**Team:** {player.get('team', 'N/A')}")
                        st.write(f"**Region:** {player.get('region', 'N/A')}")
                        st.write(f"**Category:** {player.get('playerCategory', 'N/A')}")
                        rating = player.get('rating', 'Not available')
                        st.write(f"**Rating:** {rating}")
                    
                    with col2:
                        st.write("**Agent Information**")
                        st.write("**Preferred Agents:**")
                        preferred_agents = player.get('preferredAgents', [])
                        for agent in preferred_agents:
                            st.write(f"- {agent}")
                    
                    st.write("**Key Strengths:**")
                    key_strengths = player.get('keyStrengths', [])
                    for strength in key_strengths:
                        st.write(f"- {strength}")
                    
                    st.write("**Justification:**")
                    st.write(player.get('justification', 'N/A'))
                    
                    st.markdown("---")

    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.write("Please try again with a different question.")
        import traceback
        st.write("Detailed error:")
        st.code(traceback.format_exc())

else:
    st.write("Please enter a question to see team details.")

st.sidebar.markdown("""
### About
This application displays detailed information about VALORANT esports team composition and player statistics.

### How to Use
1. Enter your question about the team in the text input
2. View the team overview and individual player details
3. Use the expanders to see detailed information about each player
""")