"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RECIPES } from "@/data/recipes";
import { PantryItem, Recommendation, mergePantryItems, scoreRecipes } from "@/lib/recommendations";
import { Camera, Loader2, Plus, SlidersHorizontal, Trash2, Upload } from "lucide-react";
import NextImage from "next/image";

const allowedVisionLabels = new Set([
  "apple",
  "banana",
  "orange",
  "carrot",
  "broccoli",
  "cucumber",
  "tomato",
  "potato",
  "lemon",
  "lime",
  "pepper",
  "bell pepper",
  "cabbage",
  "lettuce",
  "onion",
  "garlic",
  "zucchini",
  "eggplant",
  "avocado",
  "mushroom",
  "egg",
  "bread",
  "sandwich",
  "pizza",
  "cake",
  "donut",
  "bowl",
  "apple",
  "banana",
  "grapes",
  "kiwi",
  "strawberry",
  "pineapple",
  "cheese",
  "yogurt",
]);

type CocoSsdModule = typeof import("@tensorflow-models/coco-ssd");

let modelPromise: Promise<Awaited<ReturnType<CocoSsdModule["load"]>>> | null = null;

const loadVisionModel = async () => {
  if (!modelPromise) {
    modelPromise = import("@tensorflow-models/coco-ssd").then(async (mod) => {
      await import("@tensorflow/tfjs");
      return mod.load();
    });
  }
  return modelPromise;
};

const parseManualEntry = (input: string): PantryItem[] => {
  return input
    .split(/,|\n/)
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const match = entry.match(/^(\d+(?:\.\d+)?)\s*(\w+)?\s+(.*)$/);
      if (match) {
        const [, qty, unit, name] = match;
        const prettyName = unit ? `${unit} ${name}` : name;
        return {
          name: prettyName.trim(),
          quantity: parseFloat(qty),
        };
      }
      return { name: entry };
    });
};

const toTitleCase = (value: string) =>
  value
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

const formatStatus = (status: number) => Math.round(Math.min(status * 100, 100));

const IngredientChip = ({
  item,
  onRemove,
  onQuantityChange,
}: {
  item: PantryItem;
  onRemove: () => void;
  onQuantityChange: (quantity: number | undefined) => void;
}) => {
  return (
    <div className="flex items-center gap-2 rounded-full border border-zinc-200 bg-white px-4 py-2 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md">
      <span className="text-sm font-medium text-zinc-800">{toTitleCase(item.name)}</span>
      <input
        className="w-16 rounded-full border border-zinc-200 px-2 py-1 text-xs text-zinc-600 focus:border-emerald-500 focus:outline-none"
        type="number"
        min={0}
        step={0.25}
        placeholder="qty"
        value={item.quantity ?? ""}
        onChange={(event) => {
          const value = event.target.value;
          onQuantityChange(value === "" ? undefined : Number.parseFloat(value));
        }}
      />
      <button
        type="button"
        className="rounded-full border border-transparent p-1 text-zinc-400 transition hover:border-red-200 hover:bg-red-50 hover:text-red-500"
        onClick={onRemove}
        aria-label={`Remove ${item.name}`}
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
};

const RecipeCard = ({ recommendation }: { recommendation: Recommendation }) => {
  const { recipe, matchedIngredients, missingIngredients } = recommendation;
  const [servings, setServings] = useState<number>(recipe.baseServings);

  const multiplier = servings / recipe.baseServings;

  const formatQuantity = (quantity: number) => {
    if (quantity < 1) {
      return Number.parseFloat(quantity.toFixed(2));
    }
    if (Number.isInteger(quantity)) {
      return quantity;
    }
    return Number.parseFloat(quantity.toFixed(1));
  };

  return (
    <article className="rounded-3xl border border-zinc-100 bg-white/90 shadow-lg shadow-emerald-100/40 backdrop-blur transition hover:-translate-y-1 hover:shadow-emerald-200/60">
      <div className="flex flex-col gap-6 p-8">
        <div className="flex flex-col gap-2">
          <span className="text-sm font-semibold uppercase tracking-wide text-emerald-600">
            {recipe.cuisine}
          </span>
          <h3 className="text-2xl font-semibold text-zinc-900">{recipe.name}</h3>
          <p className="text-sm text-zinc-600">{recipe.description}</p>
          <div className="flex flex-wrap gap-2 pt-2">
            {recipe.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-600"
              >
                #{tag}
              </span>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-3 rounded-2xl bg-gradient-to-br from-zinc-50 to-white p-4">
          <div className="flex items-center justify-between text-sm font-medium text-zinc-600">
            <span>Dynamic ingredient calculator</span>
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <SlidersHorizontal className="h-3.5 w-3.5" />
              <span>{servings} servings</span>
            </div>
          </div>
          <input
            type="range"
            min={Math.max(1, recipe.baseServings / 2)}
            max={recipe.baseServings * 2}
            step={1}
            value={servings}
            onChange={(event) => setServings(Number.parseInt(event.target.value, 10))}
            className="accent-emerald-500"
          />
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <section className="space-y-3">
            <h4 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
              Adjusted ingredients
            </h4>
            <ul className="space-y-3 text-sm text-zinc-700">
              {recipe.ingredients.map((ingredient) => (
                <li key={ingredient.name} className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <span className="font-medium">{toTitleCase(ingredient.name)}</span>
                    {matchedIngredients.includes(ingredient.name) && (
                      <span className="ml-2 inline-flex items-center rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold uppercase text-emerald-600">
                        in pantry
                      </span>
                    )}
                  </div>
                  <span className="text-right text-zinc-500">
                    {formatQuantity(ingredient.quantity * multiplier)} {ingredient.unit}
                  </span>
                </li>
              ))}
            </ul>
          </section>

          <section className="space-y-3">
            <h4 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
              Cooking flow
            </h4>
            <ol className="space-y-3 text-sm text-zinc-700">
              {recipe.instructions.map((step, index) => (
                <li key={step} className="flex gap-3">
                  <span className="mt-1 flex h-6 w-6 items-center justify-center rounded-full bg-emerald-500 text-xs font-semibold text-white">
                    {index + 1}
                  </span>
                  <p>{step}</p>
                </li>
              ))}
            </ol>
          </section>
        </div>

        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded-full bg-emerald-50 px-3 py-1 font-medium text-emerald-600">
            Pantry matches: {matchedIngredients.length}
          </span>
          <span className="rounded-full bg-orange-50 px-3 py-1 font-medium text-orange-500">
            Missing items: {missingIngredients.length}
          </span>
        </div>
      </div>
    </article>
  );
};

const RecipeRecommender = () => {
  const [pantryItems, setPantryItems] = useState<PantryItem[]>([]);
  const [manualEntry, setManualEntry] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [visionLoading, setVisionLoading] = useState(false);
  const [visionStatus, setVisionStatus] = useState<string | null>(null);
  const [visionConfidence, setVisionConfidence] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const recommendations = useMemo(
    () => scoreRecipes(RECIPES, pantryItems),
    [pantryItems]
  );

  const handleAddPantryItems = useCallback(
    (items: PantryItem[]) => {
      if (!items.length) return;
      setPantryItems((current) => mergePantryItems([...current, ...items]));
    },
    []
  );

  const handleManualSubmit = useCallback(() => {
    const parsed = parseManualEntry(manualEntry);
    handleAddPantryItems(parsed);
    setManualEntry("");
  }, [handleAddPantryItems, manualEntry]);

  const detectIngredients = useCallback(
    async (file: File) => {
      setVisionLoading(true);
      setVisionStatus("Loading vision model…");
      try {
        const model = await loadVisionModel();
        setVisionStatus("Analyzing image…");
        const objectUrl = URL.createObjectURL(file);
        const imageElement = new window.Image();
        imageElement.src = objectUrl;
        await new Promise<void>((resolve, reject) => {
          imageElement.onload = () => resolve();
          imageElement.onerror = () => reject(new Error("Could not load image"));
        });

        const predictions = await model.detect(imageElement);
        URL.revokeObjectURL(objectUrl);

        const ingredients = predictions
          .filter((prediction) => prediction.score >= 0.4)
          .map((prediction) => ({
            score: prediction.score,
            className: prediction.class.toLowerCase(),
          }))
          .filter(({ className }) => allowedVisionLabels.has(className));

        if (!ingredients.length) {
          setVisionStatus("We could not confidently detect ingredients. Try another angle.");
          setVisionConfidence(0);
          return;
        }

        const pantry = ingredients.map(({ className, score }) => ({
          name: className,
          quantity: Number.parseFloat((score * 2).toFixed(2)),
        }));
        handleAddPantryItems(pantry);
        const averageConfidence =
          ingredients.reduce((total, ingredient) => total + ingredient.score, 0) /
          ingredients.length;
        setVisionConfidence(averageConfidence);
        setVisionStatus(
          `Detected ${ingredients.length} item${ingredients.length > 1 ? "s" : ""} with ${
            formatStatus(averageConfidence)
          }% confidence.`
        );
      } catch (error) {
        setVisionStatus(error instanceof Error ? error.message : "Vision analysis failed.");
        setVisionConfidence(0);
      } finally {
        setVisionLoading(false);
      }
    },
    [handleAddPantryItems]
  );

  const handleImageChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const [file] = event.target.files ?? [];
      if (!file) return;
      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);
      void detectIngredients(file);
    },
    [detectIngredients]
  );

  useEffect(() => {
    return () => {
      if (imagePreview) URL.revokeObjectURL(imagePreview);
    };
  }, [imagePreview]);

  const removePantryItem = useCallback((name: string) => {
    setPantryItems((items) => items.filter((item) => item.name !== name));
  }, []);

  const updatePantryQuantity = useCallback((name: string, quantity?: number) => {
    setPantryItems((items) =>
      items.map((item) => (item.name === name ? { ...item, quantity } : item))
    );
  }, []);

  const clearPantry = useCallback(() => {
    setPantryItems([]);
    setVisionStatus(null);
    setVisionConfidence(0);
    setImagePreview(null);
    setManualEntry("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const platformRecommendation =
    "A responsive web application provides instant accessibility across devices. It leverages the browser camera APIs for ingredient capture, works gracefully on desktops and mobile browsers, and ships seamlessly to Vercel for global deployment without app store friction.";

  return (
    <div className="flex flex-col gap-10">
      <section className="rounded-3xl border border-emerald-100 bg-gradient-to-br from-emerald-500 via-emerald-500 to-emerald-600 p-10 text-white shadow-xl shadow-emerald-200/30">
        <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div className="space-y-4 md:w-2/3">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-emerald-100">
              Platform Strategy
            </p>
            <h1 className="text-4xl font-semibold leading-tight">
              AI-powered recipe curation that meets people where they cook.
            </h1>
            <p className="text-base text-emerald-50">{platformRecommendation}</p>
          </div>
          <div className="rounded-2xl border border-white/20 bg-white/10 p-6 text-sm backdrop-blur">
            <p className="font-semibold uppercase tracking-wide text-emerald-50">
              Deployment notes
            </p>
            <ul className="mt-3 space-y-2 text-emerald-50">
              <li>• Runs on Next.js App Router</li>
              <li>• Client-side vision via TensorFlow.js</li>
              <li>• Works offline with cached models</li>
              <li>• Production ready for Vercel</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="flex flex-col gap-8">
          <div className="rounded-3xl border border-zinc-100 bg-white/90 p-8 shadow-lg shadow-zinc-100/60 backdrop-blur">
            <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-zinc-900">
                  Capture what&apos;s in your kitchen
                </h2>
                <p className="mt-2 text-sm text-zinc-600">
                  Upload or snap a photo of ingredients. Computer vision labels ingredients and adds
                  them straight to your pantry.
                </p>
              </div>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-2 rounded-full border border-emerald-500 bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-emerald-600"
              >
                <Camera className="h-4 w-4" />
                Use camera
              </button>
            </div>

            <div className="mt-6 grid gap-6 md:grid-cols-2">
              <label className="flex h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-3xl border-2 border-dashed border-emerald-200 bg-emerald-50/40 p-6 text-center text-sm text-emerald-600 transition hover:border-emerald-400 hover:bg-emerald-100/60">
                <Upload className="h-8 w-8" />
                <span className="font-medium">Drop a photo or browse files</span>
                <span className="text-xs text-emerald-500">
                  JPEG or PNG up to 10MB. Camera access available on mobile.
                </span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  capture="environment"
                  className="hidden"
                  onChange={handleImageChange}
                />
              </label>

              <div className="h-56 rounded-3xl border border-zinc-100 bg-zinc-50/80 p-4">
                {imagePreview ? (
                  <div className="relative h-full w-full overflow-hidden rounded-2xl">
                    <NextImage
                      src={imagePreview}
                      alt="Ingredient preview"
                      fill
                      className="object-cover"
                      sizes="(max-width: 768px) 100vw, 50vw"
                      unoptimized
                    />
                  </div>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-zinc-400">
                    Your photo preview appears here
                  </div>
                )}
              </div>
            </div>

            <div className="mt-6 flex items-center justify-between rounded-2xl border border-emerald-100 bg-emerald-50/70 px-5 py-3 text-sm text-emerald-700">
              <div className="flex items-center gap-3">
                {visionLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Camera className="h-4 w-4" />
                )}
                <p>{visionStatus ?? "Vision assistant is idle and ready."}</p>
              </div>
              <span className="rounded-full bg-white/80 px-3 py-1 text-xs font-semibold text-emerald-600">
                Confidence {formatStatus(visionConfidence)}%
              </span>
            </div>
          </div>

          <div className="rounded-3xl border border-zinc-100 bg-white/90 p-8 shadow-lg shadow-zinc-100/60 backdrop-blur">
            <h2 className="text-2xl font-semibold text-zinc-900">Add or refine ingredients</h2>
            <p className="mt-2 text-sm text-zinc-600">
              Paste groceries, fine-tune detected ingredients, or note quantities for smarter
              suggestions.
            </p>
            <div className="mt-6 flex flex-col gap-4">
              <textarea
                value={manualEntry}
                onChange={(event) => setManualEntry(event.target.value)}
                placeholder="Example: 2 cups spinach, 1 lime, 1 avocado"
                className="h-32 w-full rounded-2xl border border-zinc-200 bg-zinc-50/60 p-4 text-sm text-zinc-700 placeholder:text-zinc-400 focus:border-emerald-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-100"
              />
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={handleManualSubmit}
                  className="inline-flex items-center gap-2 rounded-full bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-emerald-600"
                >
                  <Plus className="h-4 w-4" />
                  Add ingredients
                </button>
                <button
                  type="button"
                  onClick={clearPantry}
                  className="inline-flex items-center gap-2 rounded-full border border-zinc-200 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 transition hover:border-red-100 hover:text-red-500"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Clear pantry
                </button>
              </div>
            </div>

            <div className="mt-6">
              <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                Pantry tracker
              </p>
              {pantryItems.length ? (
                <div className="mt-3 flex flex-wrap gap-3">
                  {pantryItems.map((item) => (
                    <IngredientChip
                      key={item.name}
                      item={item}
                      onRemove={() => removePantryItem(item.name)}
                      onQuantityChange={(quantity) => updatePantryQuantity(item.name, quantity)}
                    />
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-sm text-zinc-500">
                  Your pantry is empty. Add items manually or scan a photo to get started.
                </p>
              )}
            </div>
          </div>
        </div>

        <aside className="flex h-full flex-col justify-between rounded-3xl border border-zinc-100 bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-8 shadow-lg shadow-indigo-100/40">
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold text-zinc-900">Smart cooking summary</h2>
            <p className="text-sm text-zinc-600">
              Our AI combines computer vision with ingredient semantics to prioritize recipes that
              respect your pantry, minimize waste, and keep cooking fun.
            </p>
          </div>
          <div className="mt-10 space-y-6 text-sm text-zinc-600">
            <div>
              <p className="font-semibold text-zinc-800">Real-time pantry sync</p>
              <p className="mt-1">
                Edits are applied instantly—adjust quantities, remove items, or add new finds while
                cooking.
              </p>
            </div>
            <div>
              <p className="font-semibold text-zinc-800">Ingredient intelligence</p>
              <p className="mt-1">
                Probabilistic matching highlights compatible recipes and surfaces required add-ons
                so you can plan substitutions.
              </p>
            </div>
            <div>
              <p className="font-semibold text-zinc-800">Dynamic scaling</p>
              <p className="mt-1">
                Serving slider recalculates ingredient weights in real time with precision rounding.
              </p>
            </div>
          </div>
          <div className="mt-10 rounded-2xl border border-purple-200 bg-white/80 p-5 text-xs text-purple-700">
            {pantryItems.length ? (
              <p>
                {pantryItems.length} ingredient{pantryItems.length > 1 ? "s" : ""} detected. Top
                recipe match score:{" "}
                {recommendations.length
                  ? `${Math.round(recommendations[0].score * 100)}%`
                  : "n/a"}{" "}
                relevance.
              </p>
            ) : (
              <p>No data yet. Add ingredients to unlock tailored recipes.</p>
            )}
          </div>
        </aside>
      </section>

      <section className="space-y-6">
        <div className="flex flex-col gap-2">
          <h2 className="text-3xl font-semibold text-zinc-900">Personalized recipe picks</h2>
          <p className="text-sm text-zinc-600">
            Receive curated meal ideas that react to your pantry inventory and serving goals.
          </p>
        </div>
        {recommendations.length ? (
          <div className="grid gap-8 lg:grid-cols-2">
            {recommendations.slice(0, 4).map((recommendation) => (
              <RecipeCard key={recommendation.recipe.id} recommendation={recommendation} />
            ))}
          </div>
        ) : (
          <div className="rounded-3xl border border-dashed border-zinc-200 bg-white/70 p-10 text-center text-sm text-zinc-500">
            Add ingredients to reveal recipes tailored to your kitchen.
          </div>
        )}
      </section>
    </div>
  );
};

export default RecipeRecommender;
