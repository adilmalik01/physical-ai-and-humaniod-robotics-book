import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatBot from '../components/ChatBot/ChatBot';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <ChatBot />
    </>
  );
}