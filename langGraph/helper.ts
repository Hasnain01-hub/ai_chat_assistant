import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import type { Runnable } from "@langchain/core/runnables";
import type { StructuredTool } from "@langchain/core/tools";
import { Pinecone } from "@pinecone-database/pinecone";
import { Document } from "@langchain/core/documents";
import { OpenAIEmbeddings, type AzureChatOpenAI } from "@langchain/openai";
import axios from "axios";
const tunePrompt = (promptMessage: string | undefined) => {
  return (
    promptMessage?.replace(/<\/?[^>]+(>|$)/g, "\n").replace(/<[^>]*>/g, "") ??
    ""
  );
};
export type UserProfile = {
  date: string;
  email: string;
  name: string;
  role: string;
  url: string;
};
export const formatUserProfile = (profile: UserProfile) => {
  return `
    Name: ${profile.name} 
    Email: ${profile.email}
	`;
};

export const createAgent = async ({
  llm,
  systemMessage,
  tools,
  profile,
  chat_history,
  optionalParams,
}: {
  llm: AzureChatOpenAI;
  systemMessage: string;
  tools?: StructuredTool[];
  profile: UserProfile;
  chat_history: Record<string, string>[] | null;
  optionalParams?: object;
}): Promise<Runnable> => {
  const toolNames = tools?.map((tool) => tool.name).join(", ") ?? "";

  systemMessage = tunePrompt(systemMessage);

  let prompt = ChatPromptTemplate.fromMessages([
    ["system", `${systemMessage}`],
    // ["system", `user chat history ${chat_history}`],
    new MessagesPlaceholder("messages"),
  ]);

  prompt = await prompt.partial({
    tool_names: toolNames,
    user_info: formatUserProfile(profile),
  });

  if (optionalParams) {
    prompt = await prompt.partial(optionalParams);
  }

  if (tools) return prompt.pipe(llm.bindTools(tools));
  return prompt.pipe(llm);
};

export function getYoutubeVideoId(url: string): string | null {
  // Regular expression to find the video ID
  const regex = /(?<=v=)[\w-]+(?![^&\s])/;

  // Search for the pattern in the URL
  const match = url.match(regex);

  // If match found, return the video ID
  if (match) {
    return match[0];
  } else {
    return null;
  }
}

export async function initPinecone() {
  try {
    const pinecone = new Pinecone({
      // environment: process.env.PINECONE_ENV, //this is in the dashboard
      apiKey: process.env.NEXT_PUBLIC_PINECONE_API_KEY,
      controllerHostUrl: process.env.PINECONE_CONTROLLER_URL,
    });

    return pinecone;
  } catch (error) {
    console.log("error", error);
    throw new Error(
      "Failed to initialize Pinecone Client, please make sure you have the correct environment and api keys"
    );
  }
}
export async function getpinecone(message: string) {
  const client = await initPinecone();
  const index = await client.index("chatbot");
  const queryEmbedding = await new OpenAIEmbeddings({
    openAIApiKey: process.env.NEXT_PUBLIC_OPEN_API_EMBEDDING,
  }).embedQuery(message);
  let queryResponse = await index.query({
    vector: queryEmbedding,
    topK: 1,
    includeValues: true,
    includeMetadata: true,
  });
  var concatenatedPageContent;
  if (queryResponse.matches.length) {
    concatenatedPageContent = queryResponse.matches[0].metadata.pageContent;
    //  Extract and concatenate page content from matched documents
    concatenatedPageContent = queryResponse.matches
      .map((match) => match.metadata.pageContent)
      .join(" ");
  }
  return concatenatedPageContent;
}

export async function setVector(combinedContent: Document[]) {
  console.log("combinedContent", combinedContent);
  const embeddingsArrays = await new OpenAIEmbeddings({
    openAIApiKey: process.env.NEXT_PUBLIC_OPEN_API_EMBEDDING,
  }).embedDocuments(
    combinedContent.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
  );
  console.log("embeddingsArrays", embeddingsArrays);
  const client = await initPinecone();
  const pinecone_indx = await client.index("chatbot");
  const batchSize = 100;
  var batch = [];
  for (var idx = 0; idx < combinedContent.length; idx++) {
    const chunk = combinedContent[idx];
    const vector = {
      id: `${combinedContent[idx]["metadata"]["source"]}_${idx}`,
      values: embeddingsArrays[idx],
      metadata: {
        ...chunk.metadata,
        loc: JSON.stringify(chunk.metadata.loc),
        pageContent: chunk.pageContent,
        txtPath: combinedContent[idx]["metadata"]["source"],
      },
    };
    batch = [...batch, vector];
    if (batch.length === batchSize || idx === combinedContent.length - 1) {
      const result = await pinecone_indx.upsert(batch);
      // Empty the batch
      console.log("result", result);
      batch = [];
    }
  }
  return "Upsert Complete";
}

export async function queryImageToText(imageData: Buffer) {
  try {
    let header = {
      "Content-Type": "application/octet-stream",
      Authorization: `Bearer ${process.env.NEXT_PUBLIC_HUGGING_FACE}`,
    };
    let url =
      "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large";
    const response = await axios.post(url, imageData, {
      headers: header,
    });
    console.log(response.data, "response.data");
    return response.data;
  } catch (error) {
    throw new Error("Error querying image-to-text API");
  }
}
