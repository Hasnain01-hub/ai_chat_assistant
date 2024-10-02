import Image from "next/image";
import React from "react";
import ReactMarkdown from "react-markdown";
import { db, storage } from "../utils/Firebase";
import {
  getStorage,
  ref,
  uploadBytesResumable,
  getDownloadURL,
} from "firebase/storage";

const Chat = ({
  chat,
  initialMessageState,
  setInput,
  message,
  idx,
  setdata,
  data,
  handleSubmit,
  email,
  image,
  imgState,
}) => {
  console.log("message inside", chat);
  // var messageArr = message.split("```");

  const setsync = async (e, id) => {
    e.preventDefault();
    console.log("id", id);
    if (data && data.length >= 0) {
      await db
        .collection("history")
        .doc(email)
        .set({
          messages: [...data, { question: chat[id], message: chat[id + 1] }],
        });
      setdata([...data, { question: chat[id], message: chat[id + 1] }]);
    } else {
      await db
        .collection("history")
        .doc(email)
        .set({
          messages: [{ question: ques, message: message }],
        });
      setdata([...data, { question: chat[id], message: chat[id + 1] }]);
    }
  };
  return (
    <>
      <div>
        <div className="container">
          <div className="chat__list">
            <div id="chat0" className="chat__item"></div>

            <div className="chat__item active" id="chat1">
              {initialMessageState && (
                <div className="chat__box your__chat">
                  <div className="author">
                    <span>You</span>
                  </div>
                  <div className="chat">
                    <p>What is a chat bot?</p>
                  </div>
                </div>
              )}
              {initialMessageState && (
                <div className="chat__box bot__chat">
                  <div className="author">
                    <span>Bot</span>
                  </div>
                  <div className="chat">
                    <p>
                      At the most basic level, a chatbot is a computer program
                      that simulates and processes human conversation (either
                      written or spoken), allowing humans to interact with
                      digital devices as if they were communicating with a real
                      person. Chatbots can be as simple as rudimentary programs
                      that answer a simple query with a single-line response, or
                      as sophisticated as digital assistants that learn and
                      evolve to deliver increasing levels of personalization as
                      they gather and process information.
                    </p>
                  </div>
                </div>
              )}
              {initialMessageState && (
                <div className="chat__box your__chat">
                  <div className="author">
                    <span>You</span>
                  </div>
                  <div className="chat">
                    <p>How do chatbots work?</p>{" "}
                    <span>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="currentColor"
                      >
                        <path d="M5.46257 4.43262C7.21556 2.91688 9.5007 2 12 2C17.5228 2 22 6.47715 22 12C22 14.1361 21.3302 16.1158 20.1892 17.7406L17 12H20C20 7.58172 16.4183 4 12 4C9.84982 4 7.89777 4.84827 6.46023 6.22842L5.46257 4.43262ZM18.5374 19.5674C16.7844 21.0831 14.4993 22 12 22C6.47715 22 2 17.5228 2 12C2 9.86386 2.66979 7.88416 3.8108 6.25944L7 12H4C4 16.4183 7.58172 20 12 20C14.1502 20 16.1022 19.1517 17.5398 17.7716L18.5374 19.5674Z"></path>
                      </svg>
                    </span>
                  </div>
                </div>
              )}
              {initialMessageState && (
                <div className="chat__box bot__chat">
                  <div className="author">
                    <span>Bot</span>
                  </div>
                  <div className="chat">
                    <p>
                      Chatbots boost operational efficiency and bring cost
                      savings to businesses while offering convenience and added
                      services to internal employees and external customers.
                      They allow companies to easily resolve many types of
                      customer queries and issues while reducing the need for
                      human interaction.
                    </p>
                  </div>
                </div>
              )}
              {idx % 2 == 0 ? (
                <div className="chat__box your__chat">
                  <div className="author">
                    <span>You</span>
                  </div>
                  <div className="chat">
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                      }}
                    >
                      <p>{message}</p>
                      {!imgState[idx + 1] && (
                        <span>
                          <svg
                            style={{ width: "25px" }}
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24"
                            fill="currentColor"
                            onClick={(e) => setsync(e, idx)}
                          >
                            <path d="M5.46257 4.43262C7.21556 2.91688 9.5007 2 12 2C17.5228 2 22 6.47715 22 12C22 14.1361 21.3302 16.1158 20.1892 17.7406L17 12H20C20 7.58172 16.4183 4 12 4C9.84982 4 7.89777 4.84827 6.46023 6.22842L5.46257 4.43262ZM18.5374 19.5674C16.7844 21.0831 14.4993 22 12 22C6.47715 22 2 17.5228 2 12C2 9.86386 2.66979 7.88416 3.8108 6.25944L7 12H4C4 16.4183 7.58172 20 12 20C14.1502 20 16.1022 19.1517 17.5398 17.7716L18.5374 19.5674Z"></path>
                          </svg>
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="chat__box bot__chat">
                  <div className="author">
                    <span>Bot</span>
                  </div>
                  {!image ? (
                    // messageArr.map((item, index) => {
                    // return (
                    <div className="chat">
                      <ReactMarkdown>{message}</ReactMarkdown>
                    </div>
                  ) : (
                    <img
                      src={message}
                      height={300}
                      width={300}
                      style={{
                        margin: "5px auto",
                      }}
                      alt="Your Image"
                    />
                  )}
                </div>
              )}
            </div>

            <div className="chat__item" id="chat2"></div>

            <div className="chat__item" id="chat3"></div>

            <div className="chat__item" id="chat4"></div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Chat;
