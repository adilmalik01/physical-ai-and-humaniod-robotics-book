import React, { useState, KeyboardEvent } from 'react';
import clsx from 'clsx';
import styles from './ChatBot.module.css';

interface ChatInputProps {
  onSendMessage: (content: string) => void;
  isLoading: boolean;
  selectedText: string | null;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading, selectedText }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = () => {
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className={styles.chatInputContainer}>
      {selectedText && (
        <div className={styles.contextIndicator}>
          Using selected text as context
        </div>
      )}
      <div className={styles.inputWrapper}>
        <textarea
          className={styles.textInput}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={selectedText
            ? "Ask about the selected text..."
            : "Ask about the book content..."}
          disabled={isLoading}
          rows={1}
        />
        <button
          className={clsx(styles.sendButton, {
            [styles.sendButtonDisabled]: isLoading || !inputValue.trim(),
          })}
          onClick={handleSubmit}
          disabled={isLoading || !inputValue.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInput;