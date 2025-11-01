"use client";

import RecipeRecommender from "@/components/RecipeRecommender";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-zinc-50 via-white to-emerald-50 px-6 py-12 md:px-10 lg:px-16">
      <div className="mx-auto max-w-6xl">
        <RecipeRecommender />
      </div>
    </main>
  );
}
