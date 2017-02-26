import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.Charset;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * 	PUBMEDFETCH
 * 	Fetch article data (abstracts, full text) for Pubmed/PMC
 * 	Uses XML because Pubmed returns malformed JSON with eFetch for some reason
 * 
 */
public class PubmedFetch {
	
	private static String baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/";
	private static String fetchURL = "efetch.fcgi?";
		

	/* Required retrieval parameters
	String[] db = {"pubmed", "pmc"};
	pmid
	*/
	
	
	/* Optional retrieval parameters
	int retstart = 0; 				// incrementing counter if retmax > 10000
	int retmax = 10; 				// max 10000
	String[] rettype = {"uilist", "medline", "abstract"};
	String[] retmode = {"xml", "text"};
	
	 */
	
	
	private boolean largeFetch = false;
	private String[] id;
	
	private String idURL;
	
	
	/**
	 * 
	 * 
	 * 
	 */
	public PubmedFetch(String query, int numQuery) {
		
		try {
			
			id = new PubmedSearch(query, numQuery).parsePmids();
			
			if (numQuery > 500) {
				this.largeFetch = true;
			}
			
			idURL = PubmedFetch.baseURL 
					+ PubmedFetch.fetchURL
					+ "&db=pubmed"
					+ "&retmax=" + (largeFetch ? 500 : numQuery)
					+ "&id=" + String.join("+", this.id);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	} // END
	
	/**
	 * readURL method
	 * Reads URL
	 * 
	 * @param  URL as string
	 * @return URL content as string
	 * @throws MalformedURLException, IOException
	 */
	public String readURL(String url) {
		
		try {			
			BufferedReader br = new BufferedReader(
								new InputStreamReader(
								new URL(url).openConnection().getInputStream(), 
								Charset.forName("UTF-8")));
			
			StringBuilder sb = new StringBuilder();
			String line;
			
			while ((line = br.readLine()) != null) {
				sb.append(line);
			}
			
			br.close();
			System.out.println(sb.toString());
			
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
	 * 
	 * 
	 * 
	 */
	public void parse(String text, String element) {
		
		System.out.println(text);

		JsonObject jsonObject = new JsonParser().parse(text).getAsJsonObject();
		
		
		
		
		
	} // END
	
	/**
	 * 
	 * 
	 * 
	 */
	public void dump() {
		
		
		
		
		
	} // END
	


	public String getIdURL() {
		return idURL;
	}


	public static void main(String[] args) {
		
		PubmedFetch f = new PubmedFetch("cilia", 1);
		
		
		
	}
	
}
