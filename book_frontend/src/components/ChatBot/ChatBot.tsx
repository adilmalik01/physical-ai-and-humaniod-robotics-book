import React, { useState, useEffect } from 'react';
import ChatWindow from './ChatWindow';
import ChatInput from './ChatInput';
import ChatButton from './ChatButton';
import styles from './ChatBot.module.css';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date | string;
}

const ChatBot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const LOCAL_BACKEND_URL = "http://localhost:8000";
  const DEPLOY_BACKEND_URL = "https://physical-ai-and-humaniod-robotics-book-production.up.railway.app";


  // Function to handle sending a message
  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    // Add user message to the conversation
    const userMessage: Message = {
      id: Date.now().toString(),
      content: typeof content === 'string' ? content : String(content || ''),
      role: 'user',
      timestamp: new Date(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      // Prepare the request body
      const requestBody = {
        messages: newMessages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        selected_text: selectedText || null,
        max_tokens: 1000,
        temperature: 0.7
      };

      // Call the backend API
      const response = await fetch(`${DEPLOY_BACKEND_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      // Add assistant response to the conversation
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: typeof data.response === 'string' ? data.response : String(data.response || ''),
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to the conversation
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      // Clear selected text after sending
      setSelectedText(null);
    }
  };

  // Function to handle text selection
  useEffect(() => {
    const handleTextSelection = () => {
      const selectedText = window.getSelection()?.toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleTextSelection);
    return () => {
      document.removeEventListener('mouseup', handleTextSelection);
    };
  }, []);

  return (
    <>
      <ChatButton onClick={() => setIsOpen(!isOpen)} isOpen={isOpen} />
      {isOpen && (
        <div className={styles.chatContainer}>
          <div className={styles.chatWindow}>
            <ChatWindow
              messages={messages}
              isLoading={isLoading}
              selectedText={selectedText}
            />
            <ChatInput
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              selectedText={selectedText}
            />
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBot;