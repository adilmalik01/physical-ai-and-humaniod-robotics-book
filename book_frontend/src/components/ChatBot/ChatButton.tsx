import React from 'react';
import clsx from 'clsx';
import styles from './ChatBot.module.css';

interface ChatButtonProps {
  onClick: () => void;
  isOpen: boolean;
}

const ChatButton: React.FC<ChatButtonProps> = ({ onClick, isOpen }) => {
  return (
    <button
      className={clsx(styles.chatButton, {
        [styles.chatButtonOpen]: isOpen,
      })}
      onClick={onClick}
      aria-label={isOpen ? 'Close chat' : 'Open chat'}
    >
      {isOpen ? (
        <span className={styles.closeIcon}>×</span>
      ) : (
        <span className={styles.chatIcon}>💬</span>
      )}
    </button>
  );
};

export default ChatButton;