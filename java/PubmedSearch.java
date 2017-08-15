/**
 * 	PUBMEDSEARCH
 * 	Search Pubmed/PMC for term(s) to get Pmids for further use
 * 	Only useful for Pmid searches, other operations are not supported
 * 		easy to extend for other functionality
 * 	
 */


// TODO - option to include only open access articles

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.Charset;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;


public class PubmedSearch {
	
	/* Required search parameters
    private static String[] db = {"pubmed", "pmc"};
    
    */
    
	/* Optional search parameters
	String[] retmode = {"xml", "json"}; 
	String[] rettype = {"uilist", "count"};
	String[] sort = {"relevance", "pub+date"};
	String[] field = {"abstract", "title", "author"};
	String usehistory = "y";
	
	*/
	
	
	private static String baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/";
	private static String searchURL = "esearch.fcgi?";
	
	private String term; 				// formatted query	
	private String url; 				// query url
	

	/**
	 * 
	 * @param term
	 * @param numPmids 
	 * @return 
	 * @throws IOException 
	 */
	public PubmedSearch(String s, int numPmids) throws IOException {
		
		// replace inter-word space(s) with + to format query
		term = s.replaceAll("\\b +\\b", "+");
		
		
		// check for empty query
		if (term.isEmpty()) throw new IOException("Empty query!");
		
		// check for a reasonable number of numPmids
		if (numPmids < 1 || numPmids > 100000) throw new IOException("Number of pmids"
				+ "to fetch is outside reasonable limits (between 1 and 100,000)");
		
		// generate query url for eSearch
		this.url =   PubmedSearch.baseURL
				   + PubmedSearch.searchURL
				   + "&db=pubmed"
				   + "&term=" + this.term
				   + "&retmax=" + numPmids
				   + "&sort=relevance"
				   + "&retmode=json";
		System.out.println(this.url);
	} // END
	
	/**
	 * readURL method
	 * Reads URL
	 * 
	 * @return URL content as string
	 * @throws MalformedURLException, IOException
	 */
	public String readURL() {
		
		try {			
			BufferedReader br = new BufferedReader(
								new InputStreamReader(
								new URL(this.url).openConnection().getInputStream(), 
								Charset.forName("UTF-8")));
			
			StringBuilder sb = new StringBuilder();
			String line;
			
			while ((line = br.readLine()) != null) {
				sb.append(line);
			}
			
			br.close();
			// System.out.println(sb.toString());
			
			return sb.toString();
		}
			
		catch (MalformedURLException e) {
			e.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
			
		return null;
		
	} // END
	
	/**
	 * parsePmids method
	 * Parse json to retrieve Pmids
	 * 
	 *  
	 * @return Array of strings representing Pmids
	 */
	public String[] parsePmids() {
		
		String text = this.readURL();
		
		JsonObject jsonObject = new JsonParser().parse(text).getAsJsonObject();
		
		if (jsonObject != null) {
		    
			JsonObject jobject = jsonObject.getAsJsonObject("esearchresult");
		    JsonArray jarray = jobject.getAsJsonArray("idlist");
		    
		    if (jarray != null) {
		    	
		    	int arraySize = jarray.size();
		    	String[] pmids = new String[arraySize];
		    	
		    	for (int i = 0; i < arraySize; i++) {
		    		pmids[i] = jarray.get(i).getAsString();
		    	}
		    	
		    	System.out.println("\n\n------- Fetched " + arraySize + " pmids -------\n\n");
		    	return pmids;
		    }
		    
		    else {
		    	System.out.println("Empty JSON array");
		    }
		}
		
		else {
			System.out.println("Empty JSON object");
		}
		
		return null;
	
	} // END
	
	
	public String getQuery() {
		return this.term;
	}
	
	public String getUrl() {
		return this.url;
	}
	


}
