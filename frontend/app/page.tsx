"use client";
import React, { useState, useRef, useEffect } from 'react';

export default function HansardChat() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState(''); // Dynamic session ID
  const scrollRef = useRef<HTMLDivElement>(null);

  // Initialize a unique thread ID only on the client side
  useEffect(() => {
    setThreadId(`session-${Math.random().toString(36).substring(7)}`);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Function to wipe the chat and start a fresh session (clears token bloat)
  const handleNewInquiry = () => {
    setMessages([]);
    setThreadId(`session-${Math.random().toString(36).substring(7)}`);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          thread_id: threadId // Uses the unique session ID
        }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);

          setMessages((prev) => {
            const last = prev[prev.length - 1];
            const others = prev.slice(0, -1);
            return [...others, { ...last, content: last.content + chunk }];
          });
        }
      }
    } catch (err) {
      console.error("Connection error:", err);
      setMessages((prev) => {
        const others = prev.slice(0, -1);
        return [...others, { role: 'assistant', content: 'Connection error. Please check if the backend is running.' }];
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#F9F7F2] text-stone-800">
      <header className="py-4 px-6 border-b border-stone-200 bg-[#F9F7F2]/80 backdrop-blur-md sticky top-0 z-10 flex justify-between items-center font-serif">
        <h1 className="text-lg font-semibold tracking-tight text-stone-900">
          OpenChambers <span className="italic font-normal text-stone-500 text-base ml-1">Research Assistant</span>
        </h1>
        <div className="flex items-center gap-4">
          <span className="text-[10px] uppercase tracking-[0.2em] text-stone-400 font-sans font-medium hidden sm:block">
            Parliamentary Records
          </span>
          <button
            onClick={handleNewInquiry}
            className="text-[11px] text-stone-500 hover:text-stone-800 border border-stone-300 px-3 py-1 rounded-full transition-colors font-sans uppercase tracking-wider bg-white/50"
          >
            New Inquiry
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-8 md:py-12">
        <div className="max-w-3xl mx-auto space-y-10">
          {messages.length === 0 && (
            <div className="h-[40vh] flex flex-col items-center justify-center text-center space-y-4 animate-in fade-in duration-700">
              <div className="px-4 py-2 bg-stone-200 rounded-full flex items-center justify-center mb-2">
                <span className="text-stone-500 text-sm font-serif tracking-wide">OpenChambers</span>
              </div>
              <h2 className="text-2xl font-serif text-stone-700 italic">How may I assist your research today?</h2>
              <p className="text-base text-stone-400 font-sans max-w-sm leading-relaxed">
                Query parliamentary debates, MP statements and voting records.
              </p>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] px-6 py-4 shadow-sm rounded-[2rem] transition-all ${
                m.role === 'user'
                  ? 'bg-stone-200/70 text-stone-800 rounded-br-sm font-sans'
                  : 'bg-white border border-stone-200/60 text-stone-900 rounded-bl-sm font-serif'
              }`}>
                {m.role === 'assistant' && (
                  <div className="flex items-center gap-2 mb-2 opacity-50">
                    <div className="w-1.5 h-1.5 bg-stone-400 rounded-full" />
                    <span className="text-[10px] uppercase tracking-widest font-sans font-bold">Record Summary</span>
                  </div>
                )}
                <p className="whitespace-pre-wrap leading-relaxed text-[17px]">
                  {m.content || (isLoading && i === messages.length - 1 ? "..." : "")}
                </p>
              </div>
            </div>
          ))}
          <div ref={scrollRef} />
        </div>
      </main>

      <footer className="p-6">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative group">
          <input
            className="w-full pl-6 pr-16 py-4 bg-white border border-stone-200 rounded-[1.5rem] shadow-sm focus:outline-none focus:ring-1 focus:ring-stone-300 transition-all text-stone-800 placeholder-stone-400 font-sans"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Inquire about parliamentary records..."
            disabled={isLoading}
          />
          <button
            type="submit"
            className="absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center bg-stone-800 text-stone-100 rounded-full hover:bg-stone-950 disabled:bg-stone-200 disabled:text-stone-400 transition-all shadow-md"
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? (
              <div className="w-4 h-4 border-2 border-stone-400 border-t-stone-100 rounded-full animate-spin" />
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>
              </svg>
            )}
          </button>
        </form>
        <div className="max-w-3xl mx-auto text-center mt-4 uppercase tracking-tighter">
           <p className="text-[10px] text-stone-400 font-sans italic">
             Parliamentary Record API • Thread ID: {threadId}
           </p>
        </div>
      </footer>
    </div>
  );
}
