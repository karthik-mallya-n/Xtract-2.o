'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { MessageSquareText, Send, Loader2, AlertCircle, Sparkles, Bot, User, Database, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ParticleBackground from '@/components/ParticleBackground';

// ─── Types ───────────────────────────────────────────────
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

interface DatasetInfo {
  total_rows: number;
  total_columns: number;
  columns: string[];
  numeric_columns: string[];
  categorical_columns: string[];
  missing_values: Record<string, number>;
}

// ─── Markdown Renderer ──────────────────────────────────
function renderMarkdown(text: string) {
  // Process the text line by line for block elements
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let inCodeBlock = false;
  let codeBlockLines: string[] = [];
  let listItems: string[] = [];
  let listType: 'ul' | 'ol' | null = null;

  const flushList = () => {
    if (listItems.length > 0 && listType) {
      const Tag = listType;
      elements.push(
        <Tag key={`list-${elements.length}`} className={`${listType === 'ol' ? 'list-decimal' : 'list-disc'} list-inside space-y-1 my-2 text-gray-300`}>
          {listItems.map((item, i) => (
            <li key={i} dangerouslySetInnerHTML={{ __html: inlineMarkdown(item) }} />
          ))}
        </Tag>
      );
      listItems = [];
      listType = null;
    }
  };

  const inlineMarkdown = (line: string): string => {
    return line
      .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-cyan-300">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code class="bg-gray-700/60 text-cyan-300 px-1.5 py-0.5 rounded text-sm font-mono">$1</code>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-cyan-400 underline hover:text-cyan-300" target="_blank" rel="noopener">$1</a>');
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Code block toggle
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre key={`code-${elements.length}`} className="bg-gray-900/80 border border-gray-700/50 rounded-lg p-4 my-3 overflow-x-auto">
            <code className="text-green-400 text-sm font-mono whitespace-pre">{codeBlockLines.join('\n')}</code>
          </pre>
        );
        codeBlockLines = [];
        inCodeBlock = false;
      } else {
        flushList();
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockLines.push(line);
      continue;
    }

    // Empty line
    if (line.trim() === '') {
      flushList();
      continue;
    }

    // Headings
    const headingMatch = line.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      flushList();
      const level = headingMatch[1].length;
      const sizes = ['text-xl font-bold', 'text-lg font-bold', 'text-base font-semibold', 'text-sm font-semibold'];
      elements.push(
        <div key={`h-${elements.length}`} className={`${sizes[level - 1]} text-white mt-4 mb-2`}
          dangerouslySetInnerHTML={{ __html: inlineMarkdown(headingMatch[2]) }}
        />
      );
      continue;
    }

    // Ordered list
    const olMatch = line.match(/^\d+\.\s+(.*)$/);
    if (olMatch) {
      if (listType === 'ul') flushList();
      listType = 'ol';
      listItems.push(olMatch[1]);
      continue;
    }

    // Unordered list
    const ulMatch = line.match(/^[-*•]\s+(.*)$/);
    if (ulMatch) {
      if (listType === 'ol') flushList();
      listType = 'ul';
      listItems.push(ulMatch[1]);
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}$/.test(line.trim())) {
      flushList();
      elements.push(<hr key={`hr-${elements.length}`} className="border-gray-700/50 my-3" />);
      continue;
    }

    // Regular paragraph
    flushList();
    elements.push(
      <p key={`p-${elements.length}`} className="text-gray-300 my-1.5 leading-relaxed"
        dangerouslySetInnerHTML={{ __html: inlineMarkdown(line) }}
      />
    );
  }

  // Flush remaining
  flushList();
  if (inCodeBlock && codeBlockLines.length > 0) {
    elements.push(
      <pre key={`code-${elements.length}`} className="bg-gray-900/80 border border-gray-700/50 rounded-lg p-4 my-3 overflow-x-auto">
        <code className="text-green-400 text-sm font-mono whitespace-pre">{codeBlockLines.join('\n')}</code>
      </pre>
    );
  }

  return <>{elements}</>;
}

// ─── Quick-Ask Chip ─────────────────────────────────────
const quickQuestions = [
  "What are the key statistics of this dataset?",
  "Are there any missing values or data quality issues?",
  "What patterns or correlations do you see in the data?",
  "Which columns are most important for prediction?",
  "Suggest the best ML model for this dataset",
  "Summarize the distribution of each column",
];

// ─── Main Component ─────────────────────────────────────
export default function AnalysisPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSummaryLoading, setIsSummaryLoading] = useState(true);
  const [fileId, setFileId] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [error, setError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Load file and get initial AI summary
  useEffect(() => {
    const init = async () => {
      const storedFileId = localStorage.getItem('currentFileId');
      const storedFileName = localStorage.getItem('currentFileName');

      if (!storedFileId) {
        setError('No dataset uploaded. Please upload a CSV file first.');
        setIsSummaryLoading(false);
        return;
      }

      setFileId(storedFileId);
      setFileName(storedFileName || 'Dataset');

      try {
        const response = await fetch(
          `http://localhost:5000/api/analysis/summary?file_id=${storedFileId}`
        );
        const result = await response.json();

        if (result.success) {
          setDatasetInfo(result.dataset_info);

          // Add the AI summary as the first message
          setMessages([
            {
              id: 'summary',
              role: 'assistant',
              content: result.summary,
              timestamp: new Date(),
            },
          ]);
        } else {
          setError(result.error || 'Failed to analyze dataset');
        }
      } catch (err) {
        console.error('Error loading analysis:', err);
        setError('Failed to connect to the backend. Please ensure the Flask server is running.');
      } finally {
        setIsSummaryLoading(false);
      }
    };

    init();
  }, []);

  // Send a message
  const sendMessage = async (text?: string) => {
    const messageText = (text || input).trim();
    if (!messageText || isLoading || !fileId) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: messageText,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));

      const response = await fetch('http://localhost:5000/api/analysis/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageText,
          file_id: fileId,
          history,
        }),
      });

      const result = await response.json();

      if (result.success) {
        const assistantMsg: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: result.response,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMsg]);
      } else {
        const errorMsg: ChatMessage = {
          id: `error-${Date.now()}`,
          role: 'system',
          content: `⚠️ ${result.error || 'Something went wrong. Please try again.'}`,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, errorMsg]);
      }
    } catch (err) {
      console.error('Chat error:', err);
      const errorMsg: ChatMessage = {
        id: `error-${Date.now()}`,
        role: 'system',
        content: '⚠️ Failed to reach the server. Please check your connection.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    // Keep only the initial summary
    setMessages(prev => prev.filter(m => m.id === 'summary'));
  };

  // ─── No File Uploaded State ───────────────────────────
  if (error && !fileId) {
    return (
      <div className="min-h-screen relative overflow-hidden bg-gray-900 flex items-center justify-center">
        <ParticleBackground />
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
        <motion.div
          className="relative z-10 text-center max-w-md mx-auto px-6"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="p-4 rounded-2xl bg-red-900/20 border border-red-500/30 inline-block mb-6">
            <AlertCircle className="h-12 w-12 text-red-400" />
          </div>
          <h2 className="text-3xl font-black text-white mb-4">No Dataset Found</h2>
          <p className="text-gray-400 mb-8">{error}</p>
          <motion.a
            href="/upload"
            className="inline-flex items-center space-x-2 px-8 py-4 rounded-xl text-white font-bold text-lg transition-all duration-300"
            style={{
              background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
              boxShadow: '0 0 30px rgba(0, 245, 255, 0.3)',
            }}
            whileHover={{ scale: 1.05, boxShadow: '0 0 40px rgba(0, 245, 255, 0.5)' }}
            whileTap={{ scale: 0.95 }}
          >
            <Database className="h-5 w-5" />
            <span>Upload Data</span>
          </motion.a>
        </motion.div>
      </div>
    );
  }

  // ─── Loading State ────────────────────────────────────
  if (isSummaryLoading) {
    return (
      <div className="min-h-screen relative overflow-hidden bg-gray-900 flex items-center justify-center">
        <ParticleBackground />
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
        <div className="relative z-10 text-center">
          <motion.div
            animate={{ rotate: 360, scale: [1, 1.1, 1] }}
            transition={{
              rotate: { duration: 3, repeat: Infinity, ease: 'linear' },
              scale: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
            }}
            className="flex justify-center mb-8"
          >
            <Sparkles className="h-20 w-20 text-cyan-400" style={{ filter: 'drop-shadow(0 0 20px rgba(0, 245, 255, 0.5))' }} />
          </motion.div>
          <h2 className="text-3xl font-black text-white mb-4">Analyzing Your Dataset</h2>
          <p className="text-gray-400 max-w-sm mx-auto">
            Our AI is reading through your data and preparing insights…
          </p>
        </div>
      </div>
    );
  }

  // ─── Main Chat UI ─────────────────────────────────────
  return (
    <div className="min-h-screen relative overflow-hidden flex flex-col" style={{ paddingTop: '100px' }}>
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-purple-900 to-cyan-900 opacity-20" />
      <div className="absolute inset-0 geometric-pattern opacity-20" />

      <div className="relative z-10 flex flex-col flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="text-center mb-6 flex-shrink-0"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center justify-center space-x-3 mb-3">
            <motion.div
              animate={{
                boxShadow: [
                  '0 0 15px rgba(0, 245, 255, 0.3)',
                  '0 0 25px rgba(153, 69, 255, 0.4)',
                  '0 0 15px rgba(0, 245, 255, 0.3)',
                ],
              }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
              className="p-3 rounded-xl bg-cyan-900/30 border border-cyan-500/30"
            >
              <MessageSquareText className="h-8 w-8 text-cyan-400" />
            </motion.div>
            <h1 className="text-3xl sm:text-4xl font-black text-white">
              <span style={{
                background: 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                AI Analysis
              </span>
            </h1>
          </div>

          {/* Dataset badge */}
          {datasetInfo && (
            <motion.div
              className="inline-flex items-center space-x-4 glass-effect rounded-xl px-5 py-2.5 text-sm"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
            >
              <span className="flex items-center text-cyan-400">
                <Database className="h-4 w-4 mr-1.5" />
                {fileName}
              </span>
              <span className="text-gray-500">|</span>
              <span className="text-gray-400">{datasetInfo.total_rows.toLocaleString()} rows</span>
              <span className="text-gray-500">|</span>
              <span className="text-gray-400">{datasetInfo.total_columns} cols</span>
            </motion.div>
          )}
        </motion.div>

        {/* Chat Messages Area */}
        <motion.div
          className="flex-1 overflow-y-auto glass-effect rounded-2xl p-4 sm:p-6 mb-4"
          style={{ maxHeight: 'calc(100vh - 370px)', minHeight: '300px' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.3 }}
                className={`flex mb-5 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {/* Avatar for assistant */}
                {msg.role !== 'user' && (
                  <div className="flex-shrink-0 mr-3 mt-1">
                    <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                      style={{
                        background: msg.role === 'system'
                          ? 'rgba(239, 68, 68, 0.15)'
                          : 'linear-gradient(135deg, rgba(0,245,255,0.15), rgba(153,69,255,0.15))',
                        border: msg.role === 'system'
                          ? '1px solid rgba(239,68,68,0.3)'
                          : '1px solid rgba(0,245,255,0.3)',
                      }}
                    >
                      {msg.role === 'system' ? (
                        <AlertCircle className="h-4 w-4 text-red-400" />
                      ) : (
                        <Bot className="h-4 w-4 text-cyan-400" />
                      )}
                    </div>
                  </div>
                )}

                {/* Bubble */}
                <div
                  className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-5 py-3.5 text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'text-white'
                      : msg.role === 'system'
                      ? 'bg-red-900/20 border border-red-500/20 text-red-300'
                      : 'bg-gray-800/60 border border-gray-700/40 text-gray-200'
                  }`}
                  style={
                    msg.role === 'user'
                      ? {
                          background: 'linear-gradient(135deg, rgba(0,245,255,0.18) 0%, rgba(153,69,255,0.18) 100%)',
                          border: '1px solid rgba(0,245,255,0.25)',
                        }
                      : undefined
                  }
                >
                  {msg.role === 'assistant' ? renderMarkdown(msg.content) : msg.content}
                </div>

                {/* Avatar for user */}
                {msg.role === 'user' && (
                  <div className="flex-shrink-0 ml-3 mt-1">
                    <div
                      className="w-9 h-9 rounded-xl flex items-center justify-center"
                      style={{
                        background: 'linear-gradient(135deg, rgba(0,245,255,0.2), rgba(153,69,255,0.2))',
                        border: '1px solid rgba(153,69,255,0.35)',
                      }}
                    >
                      <User className="h-4 w-4 text-purple-400" />
                    </div>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Typing indicator */}
          {isLoading && (
            <motion.div
              className="flex items-center space-x-3 mb-2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, rgba(0,245,255,0.15), rgba(153,69,255,0.15))',
                  border: '1px solid rgba(0,245,255,0.3)',
                }}
              >
                <Bot className="h-4 w-4 text-cyan-400" />
              </div>
              <div className="bg-gray-800/60 border border-gray-700/40 rounded-2xl px-5 py-3.5">
                <div className="flex space-x-1.5">
                  {[0, 1, 2].map(i => (
                    <motion.div
                      key={i}
                      className="w-2 h-2 rounded-full bg-cyan-400"
                      animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1, 0.8] }}
                      transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                    />
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </motion.div>

        {/* Quick Questions */}
        {messages.length <= 1 && (
          <motion.div
            className="flex-shrink-0 mb-3 flex flex-wrap gap-2 justify-center"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            {quickQuestions.map((q, i) => (
              <motion.button
                key={i}
                onClick={() => sendMessage(q)}
                className="text-xs sm:text-sm px-3 py-1.5 rounded-lg border border-gray-700/50 text-gray-400 hover:text-cyan-400 hover:border-cyan-500/40 transition-all duration-200"
                style={{ background: 'rgba(255,255,255,0.03)' }}
                whileHover={{ scale: 1.03, y: -1 }}
                whileTap={{ scale: 0.97 }}
                disabled={isLoading}
              >
                {q}
              </motion.button>
            ))}
          </motion.div>
        )}

        {/* Input Area */}
        <motion.div
          className="flex-shrink-0 glass-effect rounded-2xl p-3 mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          style={{ border: '1px solid rgba(0,245,255,0.15)' }}
        >
          <div className="flex items-end space-x-3">
            {/* Clear chat */}
            <motion.button
              onClick={clearChat}
              className="flex-shrink-0 p-2.5 rounded-xl text-gray-500 hover:text-red-400 transition-colors duration-200"
              title="Clear conversation"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <Trash2 className="h-5 w-5" />
            </motion.button>

            {/* Text area */}
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your dataset…"
              rows={1}
              className="flex-1 bg-transparent border-0 text-white placeholder-gray-500 text-sm sm:text-base resize-none focus:outline-none focus:ring-0 py-2.5"
              style={{ maxHeight: '120px', minHeight: '40px' }}
              disabled={isLoading}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = Math.min(target.scrollHeight, 120) + 'px';
              }}
            />

            {/* Send button */}
            <motion.button
              onClick={() => sendMessage()}
              disabled={!input.trim() || isLoading}
              className="flex-shrink-0 p-2.5 rounded-xl text-white transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
              style={{
                background: input.trim() && !isLoading
                  ? 'linear-gradient(135deg, #00f5ff 0%, #9945ff 100%)'
                  : 'rgba(255,255,255,0.05)',
                boxShadow: input.trim() && !isLoading ? '0 0 20px rgba(0,245,255,0.3)' : 'none',
              }}
              whileHover={input.trim() && !isLoading ? { scale: 1.1 } : {}}
              whileTap={input.trim() && !isLoading ? { scale: 0.9 } : {}}
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </motion.button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
