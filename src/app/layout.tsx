import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/navigation";
import { ToastProvider } from "@/components/ToastProvider";

const inter = Inter({ 
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
  display: 'swap',
});

export const metadata: Metadata = {
  title: "Xtract AI - Next-Gen Machine Learning Platform",
  description: "Harness the power of AI with our cutting-edge ML platform. Upload data, get instant recommendations, and deploy models in minutes.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
      </head>
      <body className={`${inter.className} antialiased overflow-x-hidden`}>
        <ToastProvider>
          <Navigation />
          <main className="relative">
            {children}
          </main>
        </ToastProvider>
      </body>
    </html>
  );
}
