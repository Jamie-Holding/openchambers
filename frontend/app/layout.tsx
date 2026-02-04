import type { Metadata } from "next";
import { Inter, EB_Garamond } from "next/font/google";
import "./globals.css";

// Inter handles the clean, modern UI (buttons, inputs)
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

// EB Garamond provides the elegant, parchment-style serif for the records
const ebGaramond = EB_Garamond({
  variable: "--font-garamond",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OpenChambers | Parliamentary Research Assistant",
  description: "AI-powered search for UK parliamentary debates, MP statements and voting records.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${ebGaramond.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
