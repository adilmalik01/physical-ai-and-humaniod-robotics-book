import React from 'react';
import clsx from 'clsx';
import styles from './ChatBot.module.css';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  selectedText: string | null;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages, isLoading, selectedText }) => {
  return (
    <div className={styles.chatWindowContainer}>
      {/* Selected text indicator */}
      {selectedText && (
        <div className={styles.selectedTextIndicator}>
          <strong>Selected for context:</strong> "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
        </div>
      )}

      {/* Messages container */}
      <div className={styles.messagesContainer}>
        {messages.length === 0 ? (
          <div className={styles.welcomeMessage}>
            <h3>Welcome to the Physical AI & Humanoid Robotics Assistant!</h3>
            <p>Ask me anything about the book content. You can also select text on the page and ask questions about it specifically.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={clsx(
                styles.message,
                message.role === 'user' ? styles.userMessage : styles.assistantMessage
              )}
            >
              <div className={styles.messageContent}>
                {message.content}
              </div>
              <div className={styles.messageTimestamp}>
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          ))
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div className={clsx(styles.message, styles.assistantMessage)}>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatWindow;