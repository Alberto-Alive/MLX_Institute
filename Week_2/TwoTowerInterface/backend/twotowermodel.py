import random

def generate_passages(query):
    # List of sample passages to generate from
    sample_passages = [
        f"{query} is often associated with many historical and cultural contexts, providing rich insights into various fields.",
        f"The study of {query} has led to significant advancements in both theoretical and applied sciences.",
        f"{query} remains a central theme in many literary and philosophical works, inspiring countless discussions.",
        f"Many experts believe that {query} plays a vital role in understanding human behavior and social structures.",
        f"Research on {query} suggests that it has profound impacts on modern technology and innovation.",
        f"While many aspects of {query} are well understood, new discoveries continue to emerge.",
        f"{query} has become a subject of interest across different academic disciplines.",
        f"Some challenges remain in fully understanding {query}, but recent studies show promising results.",
        f"Exploring the concept of {query} reveals many fascinating interconnections with other subjects.",
        f"{query} is a multifaceted topic that spans various domains, from science to philosophy."
    ]
    
    # Randomly select five passages from the sample
    selected_passages = random.sample(sample_passages, 5)
    return [[para] for para in selected_passages]  # Wrap each passage in a list

# Example usage
# query = "artificial intelligence"
# passages = generate_passages(query)
# print(passages)
