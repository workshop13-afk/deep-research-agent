import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Deep Research Agent",
  description: "AI-powered research over local datasets",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full">{children}</body>
    </html>
  );
}
