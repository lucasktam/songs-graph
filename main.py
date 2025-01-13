import requests
import json 
import re
from dotenv import load_dotenv
import os
# Spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
GENIUS_BASE_URL = "https://api.genius.com"

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Endpoint to search for a song
id_search_endpoint = f"{GENIUS_BASE_URL}/search"

headers = {
    "Authorization": f"Bearer {GENIUS_ACCESS_TOKEN}"
}

def clean_string(string):
    return re.sub(r'\s', '', string.lower())

artist_desc_cache = {} # cache to avoid redundant API calls when we already found an artist's description

def search(q_songname, q_artist):
    # First, search for the songname + " " + artist 
    id_search_params = {
        "q": f"{q_songname} {q_artist}"  
    }
    
    id_search_response = requests.get(id_search_endpoint, headers=headers, params=id_search_params)
    song_id = -1
    # If the response is successful
    if id_search_response.status_code == 200:
        data = id_search_response.json()
        
        
        for hit in data['response']['hits']:
            artist = hit['result']['artist_names']
            # If a song is found with the correct artist, then set the correct id
            if clean_string(q_artist) in clean_string(artist) or clean_string(artist) in clean_string(q_artist):
                song_id = hit['result']['id']
                break
    else:
        print(f"Error: {id_search_response.status_code}")
        return -1

    # Else, if can't find the correct artist in list of songs
    if (song_id == -1):
        # Then we'll try searching for just the songname 
        id_search_params = {
            "q": f"{q_songname}"  
        }
        
        id_search_response = requests.get(id_search_endpoint, headers=headers, params=id_search_params)

        if id_search_response.status_code == 200:
            data = id_search_response.json()

            for hit in data['response']['hits']:
                artist = hit['result']['artist_names']
                # If a song is found with the correct artist, then set the correct id
                if clean_string(q_artist) in clean_string(artist) or clean_string(artist) in clean_string(q_artist):
                    song_id = hit['result']['id']
                    break

        else:
            print(f"Error: {id_search_response.status_code}")
            return -1

    # If we found a valid song then look for the artist's description. 
    if (song_id != -1):
        desc_endpoint = f"{GENIUS_BASE_URL}/songs/{song_id}?text_format=plain"
        desc_response = requests.get(desc_endpoint, headers=headers)

        if desc_response.status_code == 200:
            data=desc_response.json()
            # get the text's description 
            description = data['response']['song']['description']['plain']
            song_art_url = data['response']['song']['song_art_image_url']
            # Get the artist's id so we can find their description later 
            artist_id = data['response']['song']['primary_artist']['id']
            
        else:
            print(f"Error: {desc_response.status_code}")
            return -1

    # if we didn't find a valid song then return -1. 
    else:
        return -1

    if artist_id in artist_desc_cache:
        return (f"{q_songname} {q_artist} {description} {artist_desc_cache[artist_id]}", song_art_url)

    else:
        artist_desc_endpoint = f"{GENIUS_BASE_URL}/artists/{artist_id}?text_format=plain"
        artist_desc_response = requests.get(artist_desc_endpoint, headers=headers)

        if artist_desc_response.status_code == 200:
            data=artist_desc_response.json()
            
            artist_desc = data['response']['artist']['description']['plain']

            if (description == '?' and artist_desc == '?'): 
                # This is a valid song, but both of them have blank descriptions (treated as '?' in Genius)
                return -1
            
            artist_desc_cache[artist_id] = artist_desc
            return (f"{q_songname} {q_artist} {description} {artist_desc}", song_art_url)
        
        else: 
            print(f"Error: {artist_desc_response.status_code}")

    
    return -1


client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_playlist_id(playlist_link):
    return str.split(str.split(playlist_link, '/')[4], '?')[0]

PLAYLIST_LINK = "https://open.spotify.com/playlist/32LslcuMOStIUsYLejG3up?si=534b59c42c0041b6"

results = sp.playlist(get_playlist_id(PLAYLIST_LINK), fields='tracks.items.track.name,tracks.items.track.artists.name')

tuples = []


for item in results['tracks']['items']:

    track_name = item['track']['name']

    artist_names = []
    for artist in item['track']['artists']:
        artist_names.append(artist['name'])
    
    tuples.append((track_name, artist_names))

descriptions = []

graph_json = {"nodes": [], "links": []}

for song, artist in tuples:
    description = search(song, artist[0])
    if (description != -1):
        descriptions.append(description[0])

        graph_json["nodes"].append({"id": f"{song} by {artist[0]}", "url": description[1]})

import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

embeddings = model.encode(descriptions)

similarities = model.similarity(embeddings, embeddings)

# Assemble graph
SIMILARITY_THRESHOLD = 0.4

condition = similarities > SIMILARITY_THRESHOLD
indices = torch.nonzero(condition)

# A set to store unique (source, target) pairs
seen_edges = set()

for index in indices:
    row, col = index[0].item(), index[1].item()

    source = graph_json["nodes"][row]["id"]
    target = graph_json["nodes"][col]["id"]

    # Check if the edge or its reverse has already been added
    if (source, target) not in seen_edges and (target, source) not in seen_edges and target != source:
        seen_edges.add((source, target))

        graph_json["links"].append({
            "source": source,
            "target": target,
            "value": similarities[row, col].item() * 10
        })

print(json.dumps(graph_json, indent=4))

