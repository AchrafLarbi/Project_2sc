import React, { useState, useEffect } from "react";
import {
  AlertTriangle,
  Car,
  Clock,
  Gauge,
  Filter,
  Search,
  Eye,
  X,
  Camera,
} from "lucide-react";

const TrafficViolationsDashboard = () => {
  const [violations, setViolations] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [speedFilter, setSpeedFilter] = useState("all");
  const [isLoading, setIsLoading] = useState(true);
  const [selectedViolation, setSelectedViolation] = useState(null);
  const [showImageModal, setShowImageModal] = useState(false);

  const SPEED_LIMIT = 100;

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(
          "http://127.0.0.1:8888/speed_violations/violations_report.json?cb=${Date.now()}`"
        );
        if (!response.ok) {
          throw new Error("Failed to fetch violations data");
        }
        const data = await response.json();
        setViolations(data);
      } catch (error) {
        console.error("Error fetching violations:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();

    // Optional: Set up polling for real-time updates
    const interval = setInterval(loadData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getSeverityLevel = (speed) => {
    const excess = speed - SPEED_LIMIT;
    if (excess < 20) return "minor";
    if (excess < 50) return "moderate";
    return "severe";
  };

  const getSeverityColor = (speed) => {
    const severity = getSeverityLevel(speed);
    switch (severity) {
      case "minor":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "moderate":
        return "text-orange-600 bg-orange-50 border-orange-200";
      case "severe":
        return "text-red-600 bg-red-50 border-red-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const openImageModal = (violation) => {
    setSelectedViolation(violation);
    setShowImageModal(true);
  };

  const closeImageModal = () => {
    setShowImageModal(false);
    setSelectedViolation(null);
  };

  const filteredViolations = violations.filter((violation) => {
    const matchesSearch =
      violation.license_plate
        .toLowerCase()
        .includes(searchTerm.toLowerCase()) ||
      violation.vehicle_id.toString().includes(searchTerm);

    const matchesFilter =
      speedFilter === "all" ||
      (speedFilter === "minor" &&
        getSeverityLevel(violation.speed) === "minor") ||
      (speedFilter === "moderate" &&
        getSeverityLevel(violation.speed) === "moderate") ||
      (speedFilter === "severe" &&
        getSeverityLevel(violation.speed) === "severe");

    return matchesSearch && matchesFilter;
  });

  const stats = {
    total: violations.length,
    minor: violations.filter((v) => getSeverityLevel(v.speed) === "minor")
      .length,
    moderate: violations.filter((v) => getSeverityLevel(v.speed) === "moderate")
      .length,
    severe: violations.filter((v) => getSeverityLevel(v.speed) === "severe")
      .length,
    avgSpeed:
      violations.reduce((sum, v) => sum + v.speed, 0) / violations.length || 0,
  };

  if (isLoading) {
    return (
      <div className="min-h-screen w-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading traffic violations...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen w-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-lg border-b border-white/10">
        <div className="w-full px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-r from-red-500 to-orange-500 p-3 rounded-xl">
                <AlertTriangle className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">
                  Traffic Violations Monitor
                </h1>
                <p className="text-gray-300">
                  AI-Powered Speed Detection System
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2 border border-white/20">
                <div className="flex items-center space-x-2">
                  <Gauge className="h-5 w-5 text-white" />
                  <span className="text-white font-semibold">
                    Speed Limit: {SPEED_LIMIT} km/h
                  </span>
                </div>
              </div>
              <div className="bg-green-500/20 backdrop-blur-sm rounded-lg px-4 py-2 border border-green-500/30">
                <span className="text-green-300 font-semibold">
                  ● Live Monitoring
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full px-6 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-300 text-sm">Total Violations</p>
                <p className="text-3xl font-bold text-white">{stats.total}</p>
              </div>
              <Car className="h-8 w-8 text-blue-400" />
            </div>
          </div>

          <div className="bg-yellow-500/10 backdrop-blur-lg rounded-xl p-6 border border-yellow-500/30">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-yellow-300 text-sm">Minor (1 - 20 km/h)</p>
                <p className="text-3xl font-bold text-yellow-400">
                  {stats.minor}
                </p>
              </div>
              <AlertTriangle className="h-8 w-8 text-yellow-400" />
            </div>
          </div>

          <div className="bg-orange-500/10 backdrop-blur-lg rounded-xl p-6 border border-orange-500/30">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-orange-300 text-sm">
                  Moderate (20 - 50 km/h)
                </p>
                <p className="text-3xl font-bold text-orange-400">
                  {stats.moderate}
                </p>
              </div>
              <AlertTriangle className="h-8 w-8 text-orange-400" />
            </div>
          </div>

          <div className="bg-red-500/10 backdrop-blur-lg rounded-xl p-6 border border-red-500/30">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-red-300 text-sm">Severe (+ 50 km/h)</p>
                <p className="text-3xl font-bold text-red-400">
                  {stats.severe}
                </p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-400" />
            </div>
          </div>

          <div className="bg-purple-500/10 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-300 text-sm">Avg Speed</p>
                <p className="text-3xl font-bold text-purple-400">
                  {stats.avgSpeed.toFixed(0)}
                </p>
              </div>
              <Gauge className="h-8 w-8 text-purple-400" />
            </div>
          </div>
        </div>

        {/* Search and Filter Controls */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search by license plate or vehicle ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-black/20 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
            </div>
            <div className="md:w-64">
              <div className="relative">
                <Filter className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                <select
                  value={speedFilter}
                  onChange={(e) => setSpeedFilter(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-black/50 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent appearance-none"
                >
                  <option value="all">All Violations</option>
                  <option value="minor">Minor Violations</option>
                  <option value="moderate">Moderate Violations</option>
                  <option value="severe">Severe Violations</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Violations List */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 overflow-hidden">
          <div className="px-6 py-4 bg-black/20 border-b border-white/10">
            <h2 className="text-xl font-semibold text-white">
              Recent Violations ({filteredViolations.length})
            </h2>
          </div>

          <div className="divide-y divide-white/10">
            {filteredViolations.length === 0 ? (
              <div className="px-6 py-12 text-center">
                <Car className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-400 text-lg">No violations found</p>
                <p className="text-gray-500">
                  Try adjusting your search or filter criteria
                </p>
              </div>
            ) : (
              filteredViolations.map((violation) => (
                <div
                  key={violation.vehicle_id}
                  className="px-6 py-4 hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div
                        className={`px-3 py-1 rounded-full text-sm font-medium border ${getSeverityColor(
                          violation.speed
                        )}`}
                      >
                        {getSeverityLevel(violation.speed).toUpperCase()}
                      </div>
                      <div>
                        <p className="text-white font-semibold text-lg">
                          {violation.license_plate.trim()}
                        </p>
                        <p className="text-gray-400">
                          Vehicle ID: {violation.vehicle_id}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-6">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-white">
                          {Math.round(violation.speed)}
                        </p>
                        <p className="text-gray-400 text-sm">km/h</p>
                        <p className="text-red-400 text-sm">
                          +{Math.round(violation.speed - SPEED_LIMIT)} over
                        </p>
                      </div>

                      <div className="text-right">
                        <div className="flex items-center text-gray-300 mb-1">
                          <Clock className="h-4 w-4 mr-1" />
                          <span className="text-sm">
                            {formatTimestamp(violation.timestamp)}
                          </span>
                        </div>
                        <button
                          onClick={() => openImageModal(violation)}
                          className="flex items-center space-x-1 text-purple-400 hover:underline"
                        >
                          <Eye className="h-4 w-4" />
                          <span>View Details</span>
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      {showImageModal && selectedViolation && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gradient-to-br from-slate-900/95 to-slate-800/95 backdrop-blur-xl rounded-2xl shadow-2xl w-full max-w-4xl relative border border-slate-700/50 overflow-hidden">
            <div className="relative bg-gradient-to-r from-slate-800/80 to-slate-700/80 px-8 py-6 border-b border-slate-600/30">
              <button
                onClick={closeImageModal}
                className="absolute top-6 right-6 text-slate-400 hover:text-white transition-colors duration-200 p-2 hover:bg-slate-700/50 rounded-full"
              >
                <X className="h-5 w-5" />
              </button>
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-orange-500/20 rounded-lg">
                  <svg
                    className="h-6 w-6 text-orange-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"
                    />
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">
                    Violation Details
                  </h2>
                  <p className="text-slate-400 text-sm">
                    AI-Powered Speed Detection System
                  </p>
                </div>
              </div>
            </div>
            <div className="p-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-4">
                  <div className="flex items-center space-x-2 mb-4">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <h3 className="text-lg font-semibold text-white">
                      License Plate
                    </h3>
                  </div>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-xl blur-sm group-hover:blur-none transition-all duration-300"></div>
                    <div className="relative bg-slate-800/50 rounded-xl p-4 border border-slate-600/30 backdrop-blur-sm">
                      <img
                        src={`http://127.0.0.1:8888/${selectedViolation.plate_image_path}`}
                        alt="License Plate"
                        className="w-full h-auto rounded-lg shadow-lg border border-slate-600/20"
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center space-x-2 mb-4">
                    <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                    <h3 className="text-lg font-semibold text-white">
                      Vehicle
                    </h3>
                  </div>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl blur-sm group-hover:blur-none transition-all duration-300"></div>
                    <div className="relative bg-slate-800/50 rounded-xl p-4 border border-slate-600/30 backdrop-blur-sm">
                      <img
                        src={`http://127.0.0.1:8888/${selectedViolation.vehicle_image_path}`}
                        alt="Vehicle"
                        className="w-full h-auto rounded-lg shadow-lg border border-slate-600/20"
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-8 pt-6 border-t border-slate-600/30">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-center space-x-6">
                    {selectedViolation.speed && (
                      <div className="flex items-center space-x-2">
                        <div className="p-2 bg-red-500/20 rounded-lg">
                          <svg
                            className="h-4 w-4 text-red-400"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M13 10V3L4 14h7v7l9-11h-7z"
                            />
                          </svg>
                        </div>
                        <div>
                          <p className="text-xs text-slate-400 uppercase tracking-wide">
                            Speed
                          </p>
                          <p className="text-white font-semibold">
                            {selectedViolation.speed} km/h
                          </p>
                        </div>
                      </div>
                    )}
                    {selectedViolation.timestamp && (
                      <div className="flex items-center space-x-2">
                        <div className="p-2 bg-green-500/20 rounded-lg">
                          <svg
                            className="h-4 w-4 text-green-400"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                        </div>
                        <div>
                          <p className="text-xs text-slate-400 uppercase tracking-wide">
                            Time
                          </p>
                          <p className="text-white font-semibold">
                            {selectedViolation.timestamp}
                          </p>
                        </div>
                      </div>
                    )}
                    {selectedViolation.location && (
                      <div className="flex items-center space-x-2">
                        <div className="p-2 bg-yellow-500/20 rounded-lg">
                          <svg
                            className="h-4 w-4 text-yellow-400"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                            />
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                            />
                          </svg>
                        </div>
                        <div>
                          <p className="text-xs text-slate-400 uppercase tracking-wide">
                            Location
                          </p>
                          <p className="text-white font-semibold">
                            {selectedViolation.location}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrafficViolationsDashboard;
