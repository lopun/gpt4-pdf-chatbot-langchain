import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import {
  ConversationalRetrievalQAChain,
  RetrievalQAChain,
} from 'langchain/chains';
import memoize from 'memoizee';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a professional Software Engineer. Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
Helpful answer in markdown format(response only in code format. no description):`;

const model = new OpenAI({
  temperature: 0, // increase temepreature to get more creative answers
  modelName: 'gpt-4', //change this to gpt-4 if you have access
});

export const makeChain = memoize((vectorStore: PineconeStore) => {
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      // questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );

  return chain;
});
