import Link from "next/link";
import { ArrowRight, Brain } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-20">
        <div className="text-center">
          <div className="flex justify-center mb-8">
            <Brain className="h-16 w-16 text-blue-600" />
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Machine Learning Made Simple
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Upload your dataset, get AI-powered model recommendations, and make predictions.
          </p>
          <Link href="/upload" className="inline-flex items-center px-8 py-4 text-lg font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg">
            Get Started
            <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
        </div>
      </div>
    </div>
  );
}
