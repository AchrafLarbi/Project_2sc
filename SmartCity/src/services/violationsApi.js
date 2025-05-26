import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8888";

export const violationsApi = {
  // Get all violations
  getAllViolations: async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/speed_violations/violations_report.json`
      );
      return response.data;
    } catch (error) {
      console.error("Error fetching violations:", error);
      throw error;
    }
  },

  // Get violation by ID
  getViolationById: async (id) => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/speed_violations/violations_report.json`
      );
      const violations = response.data;
      return violations.find((v) => v.vehicle_id === id);
    } catch (error) {
      console.error("Error fetching violation by ID:", error);
      throw error;
    }
  },

  // Get violations by speed range
  getViolationsBySpeedRange: async (minSpeed, maxSpeed) => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/speed_violations/violations_report.json`
      );
      const violations = response.data;
      return violations.filter(
        (v) => v.speed >= minSpeed && v.speed <= maxSpeed
      );
    } catch (error) {
      console.error("Error fetching violations by speed range:", error);
      throw error;
    }
  },

  // Get violations by date range
  getViolationsByDateRange: async (startDate, endDate) => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/speed_violations/violations_report.json`
      );
      const violations = response.data;
      return violations.filter((v) => {
        const violationDate = new Date(v.timestamp);
        return violationDate >= startDate && violationDate <= endDate;
      });
    } catch (error) {
      console.error("Error fetching violations by date range:", error);
      throw error;
    }
  },
};
