// Usage: node index.js
import dotenv from 'dotenv';
dotenv.config();
// Import the required libraries
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HarmBlockThreshold, HarmCategory } from "@google/generative-ai";
import natural from 'natural';
import { TextLoader } from "langchain/document_loaders/fs/text";
// Create a new instance of the ChatGoogleGenerativeAI model
const model = new ChatGoogleGenerativeAI({
  model: "gemini-pro",
  maxOutputTokens: 2048,
  safetySettings: [
    {
      category: HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
  ],
});
// create a new instance of the WordTokenizer
const tokenizer = new natural.WordTokenizer();
// Define the function to load the documents
async function loadDocuments() {
    const loader = new TextLoader("./data/info.txt");
    const docs = await loader.load();
    console.log("Loaded documents:", docs);
    return docs;
}
// Define the function to retrieve the data
async function retrieveData(query, documents) {
  const queryTokens = tokenizer.tokenize(query.toLowerCase());
  let bestMatch = null;
  let highestScore = 0;
  // Iterate over the documents and find the best match
  documents.forEach(document => {
    if (!document || !document.pageContent) {
        console.error("Document or document page content is undefined.");
        return;
    }
    const sentences = document.pageContent.split(/[\r\n]+/).filter(line => line.trim() !== ''); // Split into sentences
    sentences.forEach(sentence => {
      console.log("Processing sentence:", sentence);
      let sentenceTokens = tokenizer.tokenize(sentence.toLowerCase());
      let intersection = sentenceTokens.filter(token => queryTokens.includes(token));
      let score = intersection.length;

      if (score > highestScore) {
        highestScore = score;
        bestMatch = sentence;
      }
    });
  });
  // Return the best match
  return bestMatch || "No relevant information found.";
}
// Define the function to generate the response
async function generateResponse(query) {
  try {
    const documents = await loadDocuments();
    if (!documents.length) {
      return "Failed to load documents or documents are empty.";
    }
    // Retrieve the data from the documents
    const retrievedData = await retrieveData(query, documents);
    // Augment the query with the retrieved data
    const augmentedQuery = retrievedData ? `${query} Considering the following fact: ${retrievedData}` : query;
    // Generate the response using the model
    const responses = await model.invoke([
      [
        "human",
        augmentedQuery
      ],
    ]);

    return responses;
  } catch (error) {
    console.error("Error during data retrieval or generation:", error);
    return "An error occurred while generating the response.";
  }
}
// Generate the response for the query
generateResponse("What is the estimated population of the Earth?")
  .then(response => console.log("Generated Response:", response.content))
  .catch(error => console.error(error));
