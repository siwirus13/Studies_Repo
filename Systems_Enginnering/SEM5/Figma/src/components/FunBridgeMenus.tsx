import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, Trophy, Settings, User, ChevronLeft, ChevronRight, Bell } from "lucide-react";

type Screen = "home" | "tournament" | "settings";

export default function FunBridgeMenus() {
  const [screen, setScreen] = useState<Screen>("home");

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white flex items-center justify-center p-6">
      <div className="w-[390px] bg-white rounded-2xl shadow-xl overflow-hidden ring-1 ring-slate-200">
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <button className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200">
              <Menu size={18} />
            </button>
            <div>
              <h3 className="text-sm font-semibold">FunBridge</h3>
              <p className="text-xs text-slate-500">Welcome back</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="relative p-2 rounded-lg bg-slate-50 hover:bg-slate-100">
              <Bell size={16} />
              <span className="absolute -top-1 -right-1 inline-flex items-center justify-center h-4 w-4 rounded-full bg-rose-500 text-white text-xs">
                3
              </span>
            </button>
            <div className="h-9 w-9 rounded-full bg-gradient-to-tr from-indigo-500 to-emerald-400 flex items-center justify-center text-white font-medium">
              AB
            </div>
          </div>
        </div>

        {/* Content area */}
        <div className="p-4">
          <AnimatePresence mode="wait" initial={false}>
            {screen === "home" && (
              <motion.div
                key="home"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.25 }}
              >
                <HomeMenu onNavigate={setScreen} />
              </motion.div>
            )}

            {screen === "tournament" && (
              <motion.div
                key="tournament"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.25 }}
              >
                <TournamentMenu onNavigate={setScreen} />
              </motion.div>
            )}

            {screen === "settings" && (
              <motion.div
                key="settings"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
              >
                <SettingsMenu onNavigate={setScreen} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Bottom tab bar */}
        <div className="border-t border-slate-100 px-4 py-3 flex items-center justify-between">
          <NavButton active={screen === "home"} onClick={() => setScreen("home")} icon={<User size={16} />} label="Play" />
          <NavButton active={screen === "tournament"} onClick={() => setScreen("tournament")} icon={<Trophy size={16} />} label="Tournaments" />
          <NavButton active={screen === "settings"} onClick={() => setScreen("settings")} icon={<Settings size={16} />} label="Settings" />
        </div>
      </div>
    </div>
  );
}

function NavButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex flex-col items-center gap-1 text-xs py-1 rounded-md hover:bg-slate-50 ${
        active ? "text-slate-900 font-semibold" : "text-slate-500"
      }`}
    >
      <div className={`p-2 rounded-md ${active ? "bg-slate-100" : ""}`}>{icon}</div>
      <span>{label}</span>
    </button>
  );
}

/* ------------------------------- SCREENS ------------------------------- */

function HomeMenu({ onNavigate }: { onNavigate: (s: Screen) => void }) {
  return (
    <div>
      <section className="mb-4">
        <h4 className="text-sm font-semibold">Quick Play</h4>
        <p className="text-xs text-slate-500">Start a new friendly or ranked game</p>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <button className="col-span-1 p-3 rounded-xl border border-slate-100 text-left hover:shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium">Friendly Match</div>
                <div className="text-xs text-slate-500">Invite friends or play instantly</div>
              </div>
              <ChevronRight size={18} />
            </div>
          </button>

          <button className="col-span-1 p-3 rounded-xl border border-slate-100 text-left hover:shadow-sm bg-gradient-to-tr from-amber-50 to-white">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium">Ranked</div>
                <div className="text-xs text-slate-500">Climb the leaderboard</div>
              </div>
              <div className="text-sm font-semibold text-amber-600">Play</div>
            </div>
          </button>
        </div>
      </section>

      <section>
        <h4 className="text-sm font-semibold">Live Tournaments</h4>
        <div className="mt-2 space-y-3">
          <MenuCard title="Daily Swiss" subtitle="Starts in 2h" badge="€1" onClick={() => onNavigate("tournament")} />
          <MenuCard title="Weekend Cup" subtitle="Ongoing" badge="Top" onClick={() => onNavigate("tournament")} />
        </div>
      </section>
    </div>
  );
}

function TournamentMenu({ onNavigate }: { onNavigate: (s: Screen) => void }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className="text-sm font-semibold">Tournaments</h4>
          <p className="text-xs text-slate-500">Join scheduled and live events</p>
        </div>
        <div className="text-xs text-slate-400">Filter</div>
      </div>

      <div className="space-y-3">
        <TournamentItem name="March Masters" time="Today — 18:00" prize="€500" />
        <TournamentItem name="Novice Open" time="Tomorrow — 14:00" prize="€50" />
        <TournamentItem name="Legends Trophy" time="Oct 30 — 20:00" prize="€1200" />
      </div>

      <div className="mt-5">
        <button onClick={() => onNavigate("home")} className="w-full py-2 rounded-xl border border-slate-100">
          <div className="flex items-center justify-center gap-2">
            <ChevronLeft size={16} /> Back
          </div>
        </button>
      </div>
    </div>
  );
}

function SettingsMenu({ onNavigate }: { onNavigate: (s: Screen) => void }) {
  return (
    <div>
      <h4 className="text-sm font-semibold mb-2">Settings</h4>
      <p className="text-xs text-slate-500 mb-4">Customize your experience</p>

      <div className="space-y-3">
        <SettingRow label="Profile" desc="Edit your name & avatar" onClick={() => {}} />
        <SettingRow label="Notifications" desc="Manage alerts" onClick={() => {}} />
        <SettingRow label="Appearance" desc="Theme & layout" onClick={() => {}} />
      </div>

      <div className="mt-6">
        <button onClick={() => onNavigate("home")} className="w-full py-2 rounded-xl bg-slate-900 text-white">
          Save & Return
        </button>
      </div>
    </div>
  );
}

/* ----------------------------- SMALL COMPONENTS ----------------------------- */

function MenuCard({
  title,
  subtitle,
  badge,
  onClick,
}: {
  title: string;
  subtitle?: string;
  badge?: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full p-3 rounded-xl border border-slate-100 text-left flex items-center justify-between hover:shadow-sm"
    >
      <div>
        <div className="text-sm font-medium">{title}</div>
        {subtitle && <div className="text-xs text-slate-500">{subtitle}</div>}
      </div>
      {badge && <div className="text-xs font-semibold text-emerald-600">{badge}</div>}
    </button>
  );
}

function TournamentItem({ name, time, prize }: { name: string; time: string; prize: string }) {
  return (
    <div className="p-3 rounded-xl border border-slate-100 flex items-center justify-between hover:bg-slate-50">
      <div>
        <div className="text-sm font-medium">{name}</div>
        <div className="text-xs text-slate-500">{time}</div>
      </div>
      <div className="text-sm font-semibold">{prize}</div>
    </div>
  );
}

function SettingRow({ label, desc, onClick }: { label: string; desc?: string; onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-full p-3 rounded-xl border border-slate-100 text-left flex items-center justify-between hover:bg-slate-50"
    >
      <div>
        <div className="text-sm font-medium">{label}</div>
        {desc && <div className="text-xs text-slate-500">{desc}</div>}
      </div>
      <ChevronRight size={16} />
    </button>
  );
}
