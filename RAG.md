The [new course](https://learn.deeplearning.ai/advanced-retrieval-for-ai/) from DeepLearning.AI with Chroma brings together various experiences from the field that could be used to improve the performance of Retrieval Augmented Generation (RAG) systems. Although the course was focused on using [Chroma](https://www.trychroma.com/) as a vector data store for the indexing and retrieval part and how it could be optimized for RAG use cases, I think most of the concepts would generalize to other vector data stores that are used in the enterprise. I would also add some of my experiences with [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) and how some of the concepts that are discussed with respect to Chroma DB, [Sentence Transformers](https://www.sbert.net/), [OpenAI](https://openai.com/) would translate to Azure AI search and [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service).

The running example in the course was to ask questions from a PDF file containing Microsoft’s annual report for Shareholders for the year 2022. This was similar to an “Ask your PDF” use case where the index size is generally limited to the size of a single file. Throughout the course Chroma was used as a Python “package” which ran in the same process/session where the core application that processed the user query and generated answers ran.

However, in the enterprise one would use a [client-server](https://docs.trychroma.com/usage-guide#running-chroma-in-clientserver-mode) based implementation of a database in order to reach production so that better scalability, operations, security, etc. could be achieved. Azure AI search is generally interfaced by a [Python](https://pypi.org/project/azure-search-documents/) (and few other officially supported languages) wrapper over its [REST API](https://learn.microsoft.com/en-us/rest/api/searchservice/) which allows client applications to perform Create, Read, Update and Delete (CRUD) operations on the data in the database.

Azure AI search’s ([previously Azure search, Azure Cognitive search](https://en.wikipedia.org/wiki/Azure_Cognitive_Search)) initial purpose was more like a Microsoft developed managed version of the [ElasticSearch](https://www.elastic.co/elasticsearch) service on Azure but has now rebranded itself as a combination of text and vector database thereby enabling “[hybrid](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)” search at scale.

## Overview of Embeddings-based retrieval

![](https://miro.medium.com/v2/resize:fit:1400/1*MRJBRkaBVo-PD-LRR1lRpA.png)

General RAG system diagram (source: Course video)

[According to Microsoft,](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview#why-choose-hybrid-search) a hybrid approach which considers the full text along with the embeddings generally performs “better” than just an embeddings based approach. Chroma [also allows to perform full-text search](https://docs.trychroma.com/usage-guide#querying-a-collection) with the embeddings search and would probably in the near future advertise itself as a “hybrid” database.

> Hybrid search combines the strengths of vector search and keyword search. The advantage of vector search is finding information that’s conceptually similar to your search query, even if there are no keyword matches in the inverted index. The advantage of keyword or full text search is precision, with the ability to apply semantic ranking that improves the quality of the initial results. Some scenarios — such as querying over product codes, highly specialized jargon, dates, and people’s names — can perform better with keyword search because it can identify exact matches.
> 
> Benchmark testing on real-world and benchmark datasets indicates that hybrid retrieval with semantic ranking offers significant benefits in search relevance.

![](https://miro.medium.com/v2/resize:fit:1400/1*CiQeeikLdJKOXsFDdtxPPw.png)

Sentence Transformer to generate embeddings (source: Course video)

In the course, the [Sentence Transformer package](https://www.sbert.net/) is used to generate embeddings using the [all-MiniLM-L6-v2 model](https://docs.trychroma.com/embeddings) locally. The dimensions of the embeddings vector are approximately 350. In the enterprise, Azure OpenAI’s text-embedding-ada-002 model is generally preferred due to the pay per use mode and the minimal effort required to “operate” the models. The models execute on Azure infrastructure as a managed service. The dimensions of each embedding vector are about 1550.

The higher dimension has a trade-off that it requires more space for storage in Azure AI search and more compute for operations on the embeddings vectors but it captures more context of the underlying text chunk which would generally mean better relevance during search.

## Pitfalls of retrieval — when simple vector search fails!

The course uses dimension reduction using the [UMAP](https://umap-learn.readthedocs.io/en/latest/) (**U**niform **M**anifold **A**pproximation and **P**rojection for Dimension Reduction) method to reduce the dimensionality of the embeddings vector but at the same time not lose too much of the spatial distance among each individual chunk represented as a point in a 2D plane compared to other dimension reduction techniques like [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis). This allows to visualize the query and the retrieved chunks along with all the existing chunks and so analyze how a query performs with respect to the chunks that it retrieves from the universe of all chunks.

![](https://miro.medium.com/v2/resize:fit:1180/1*bMPQ-t3ZA_FWnwYpA5GDtQ.png)

Chunks found nearest to a query that is relevant to the document (Source: course video)

![](https://miro.medium.com/v2/resize:fit:1400/1*I1XkTdWSUJBFY4bL3ySyLQ.png)

Chunks found nearest to a query that is irrelevant to the document — distractors for the LLM (Source: course video)

The main issue is that the embedding model used to embed the query and the data does not have any knowledge of the task or the query we are trying to answer when we retrieve the information. The retrieval system performs suboptimally since we ask it to perform a specific task using a general representation.

## Query Expansion

The main idea here is to use powerful LLMs to enhance and augment the queries that are sent to the vector based retrieval systems to get better results.

1. [Expansion with generated/hallucinated answer](https://arxiv.org/abs/2305.03653)

![](https://miro.medium.com/v2/resize:fit:1334/1*FWCdwy3PSTpdXizNErCadg.png)

Query expansion with generated answers (Soure: Course video)

def augment_query_generated(query, model="gpt-3.5-turbo"):  
    messages = [  
        {  
            "role": "system",  
            "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report. "  
        },  
        {"role": "user", "content": query}  
    ]   
  
    response = openai_client.chat.completions.create(  
        model=model,  
        messages=messages,  
    )  
    content = response.choices[0].message.content  
    return content

The above is a sample system prompt as shown in the course to augment the user’s query with a hallucinated answer that will serve as a better context to search for relevant documents in the vector database.

original_query = "Was there significant turnover in the executive team?"  
hypothetical_answer = augment_query_generated(original_query)  
joint_query = f"{original_query} {hypothetical_answer}"  
results = chroma_collection.query(query_texts=joint_query, n_results=5, include=['documents', 'embeddings'])

The above snippet shows how to leverage the hypothetical answer to augment the query that is sent to the vector database.

![](https://miro.medium.com/v2/resize:fit:1400/1*ZUalSGkCSPLUw14ztj5r4w.png)

Query moved from red to orange in space hopefully producing better results (Source: Course video)

2. Expansion with multiple queries

![](https://miro.medium.com/v2/resize:fit:1336/1*Tfhjiyc3qSYVPFxWZfDJFA.png)

Expansion with multiple queries (source: course video)

def augment_multiple_query(query, model="gpt-3.5-turbo"):  
    messages = [  
        {  
            "role": "system",  
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about an annual report. "  
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "  
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."  
            "Make sure they are complete questions, and that they are related to the original question."  
            "Output one question per line. Do not number the questions."  
        },  
        {"role": "user", "content": query}  
    ]  
  
    response = openai_client.chat.completions.create(  
        model=model,  
        messages=messages,  
    )  
    content = response.choices[0].message.content  
    content = content.split("\n")  
    return content

The above is a sample system prompt as shown in the course to augment the user’s query with multiple queries that are related to the original query. Prompt engineering becomes important as the output in both the cases depends on the prompts that are used.

original_query = "What were the most important factors that contributed to increases in revenue?"  
augmented_queries = augment_multiple_query(original_query)  
queries = [original_query] + augmented_queries  
results = chroma_collection.query(query_texts=queries, n_results=5, include=['documents', 'embeddings'])

The above snippet as taken from the course show how a batch of queries are used to retrieve chunks in a single batch as Chroma can handle multiple queries. However, the results will have some overlap as the same document can be returned for multiple queries and so an additional deduplication step is required after the query step.

![](https://miro.medium.com/v2/resize:fit:1400/1*S9LaQN6Lv3AORgyLJXkzgA.png)

More and different types of information can be retrieved for complex queries (source: Course video)

There are a few downsides to this technique:

1. Since we issue multiple requests to the information retrieval system and need subsequent deduplication, latency becomes a concern for chat based systems
2. Also, there are a lot more results due to the multiple queries and not all of them will be relevant to the original user question. So an additional step of cross-encoder re-ranking is required to “re-rank” the results. This can potentially return more relevant results at the cost of higher latency and operational costs to host and maintain the re-ranking model.

## Cross-encoder re-ranking for the “long tail”

Cross-encoder reranker is used to score the relevancy of the retrieved results for the query that is provided by the user.

![](https://miro.medium.com/v2/resize:fit:1400/1*4q3JtUpIhzezfoHmxNFTww.png)

ReRanking conceptual diagram (source: Course video)

Reranker tries to solve the problem of finding which of the retrieved results are the most relevant to the current query instead of just being the nearest neighbors in the embedding space. The reranker that is used in the course was the pre-trained [cross-encoder/ms-marco-MiniLM-L-6-v2 model](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html) using the Sentence Transformers library.

![](https://miro.medium.com/v2/resize:fit:1338/1*jVMgs3F1AhVuJka37jkHYA.png)

Bi encoders vs cross-encoders in the Sentence Transformers library (source: Course video)

The cross-encoder model is passed a list of query and retrieved document pair and it returns a list of scores for each pair. The results are then ordered in descending order by the scores and top 10 results are only sent as context to the LLM.

The benefit of the cross encoder approach is to bring in relevant documents from the long tail into the context window as it might be more relevant for the query than the results that are just the top 5/10 nearest neighbors of the query in the embeddings space.

This can be used with the results from query expansion to obtain the **most** relevant documents to be passed to the LLM’s context rather than passing all the documents which can be distracting for the LLM. Cross encoder models are generally lightweight and could run in the same process as the chat application.

The reranker scores can change based on the query even when the results that the retrieval system provides are the same since the reranker can emphasise different parts of the query than the embedding model does and so the ranking that it provides is more conditioned on the specific query than just what is returned by the embeddings based retrieval system itself.

## Embedding adaptors

This approach uses user feedback about the relevancy of retrieved results to automatically improve the performance of the retrieval system. It alters the embedding of a query directly to produce better retrieval results. The embedding adaptor is a light weight model trained on feedback data about the relevancy of retrieved results for a set of queries.

![](https://miro.medium.com/v2/resize:fit:1338/1*fGucwvymAzhUgRaAK7Q89Q.png)

Additional step of “Embedding adaptor” to better adapt the embeddings for the query (Source: Course video)

![](https://miro.medium.com/v2/resize:fit:1192/1*ogxo7q3zDASzfL5WXvccow.png)

Adapted queries in embedding space (Source: Course video)

This approach relies on feedback either from users or from synthetic data generated using LLMs and helps to customize query embeddings for the specific application. User feedback data is preferred for this approach.

## Other techniques in the research literature

![](https://miro.medium.com/v2/resize:fit:1400/1*HXRMKh8ur0FC0Xc_qbJmug.png)

Techniques in research literature on improving retrival (Source: Course video)

All the “deep” methods imply training deep learning models including transformer models that are specialized for the task.